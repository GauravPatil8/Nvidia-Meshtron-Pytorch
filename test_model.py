from meshtron.model import Meshtron
from meshtron.encoder_conditioning import ConditioningEncoder
from meshtron.VertexTokenizer import VertexTokenizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from pipeline.stages.inference import Inference
from pipeline.config import ConfigurationManager
from pipeline.utils.data import get_mesh_stats, normalize_mesh_to_bbox, add_gaussian_noise, set_zero_vector, write_obj
from pipeline.utils.model import get_weights_path
from pipeline.utils.common import get_root_folder
from tqdm import tqdm
import torch
import torch.nn as nn
import trimesh
import os
# --------------------------------------------------------------------------------------------
#  This script is used to verify that the model runs end-to-end without errors.
#  It uses randomly generated data to test forward and backward passes,
#  but does NOT compute a meaningful loss or perform real training.
#
#  Purpose: Debug and ensure the model architecture, attention modules,
#  and gradient flow work correctly before actual training.
# --------------------------------------------------------------------------------------------

torch.manual_seed(123)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.01
print(DEVICE)
tokenizer = VertexTokenizer(128, 1.0)


NUM_EPOCHS = 5
NUM_SAMPLES = 20
SEQ_LEN = 9
sub_input = torch.randint(0,131,[NUM_SAMPLES,SEQ_LEN], dtype=torch.int64)
INPUT_DATA = torch.cat([torch.tensor([[tokenizer.SOS] * 9] * NUM_SAMPLES, dtype=torch.int64), sub_input], dim=-1).to(device=DEVICE)
POINT_CLOUD = torch.randn([NUM_SAMPLES,128,6], dtype=torch.float16).to(device=DEVICE)
QUAD_RATIO = torch.rand(NUM_SAMPLES,1).to(device=DEVICE, dtype=torch.float16)
FACE_COUNT = torch.randint(5000, 15000, (NUM_SAMPLES,1)).to(device=DEVICE, dtype=torch.float16)
TARGET = torch.cat([sub_input, torch.tensor([[tokenizer.EOS] * 9] * NUM_SAMPLES, dtype = torch.int64)],dim=-1).to(device=DEVICE)
MASK = torch.tril(torch.ones((1, SEQ_LEN+9, SEQ_LEN+9))).type(torch.bool).to(device=DEVICE)
SCALER = torch.amp.GradScaler()
def get_model():

    encoder = ConditioningEncoder(
                                input_channels= 6, 
                                input_axis= 1,
                                num_freq_bands=6,
                                max_freq=10.,
                                depth=8,
                                num_latents=24,
                                latent_dim=24,
                                cross_heads=1,
                                latent_heads=2,
                                cross_dim_head=12,
                                latent_dim_head=12,
                                num_classes=1,
                                attn_dropout=0.2,
                                ff_dropout=0.1,
                                weight_tie_layers=6,
                                fourier_encode_data=True,
                                self_per_cross_attn=2,
                                final_classifier_head=False,
                                dim_ffn=12
                                )
    return Meshtron(dim = 24, 
                    embedding_size= 131,
                    n_heads=2,
                    head_dim=12,
                    window_size= 3,
                    d_ff=12,
                    hierarchy="4@1 8@3 12@9 8@3 4@1",
                    dropout=0.2,
                    pad_token=tokenizer.PAD.item(),
                    condition_every_n_layers=4,
                    encoder=encoder,
                    ).to(device=DEVICE)

def train(model: Meshtron, tokenizer: VertexTokenizer):
    model.train()
    g_step = 0
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, eps=1e-9, weight_decay=1e-2)
    scaler = torch.amp.GradScaler()
    loss_hist = list()
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=85,  # total_iters - warmup_iters
        eta_min=0.003
    )

    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD.item(), label_smoothing=0.0).to(DEVICE)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(NUM_EPOCHS):
        iter = tqdm(range(NUM_SAMPLES), desc=f"Processing epoch: {epoch+1:02d}")
        for i in iter:
            #forward
            with torch.amp.autocast(device_type='cuda'):
                output = model(INPUT_DATA[i].unsqueeze(0), POINT_CLOUD[i].unsqueeze(-3), FACE_COUNT[i], QUAD_RATIO[i], MASK)
                out_prob = model.project(output)
                # print(out_prob)
                loss = loss_func(out_prob.view(-1, tokenizer.vocab_size), TARGET[i].view(-1))
            iter.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss_hist.append(f"{loss.item():6.3f}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            g_step+=1
    with open(os.path.join("/root/meshtron/Nvidia-Meshtron-Pytorch/pipeline/logs","loss.txt"),'w') as f:
            for loss_abc in loss_hist:
                f.write(f"{loss_abc}\n")
    return loss.item(), g_step

def main():
    model = get_model()

    loss, total_steps = train(model, tokenizer)
    print(f"Model Trained successfully \n-The total steps are: {total_steps} \n-Loss: {loss}")


def get_point_cloud_data(mesh_path: str):
    mesh = trimesh.load_mesh(mesh_path, file_type = 'obj')
    vertices = normalize_mesh_to_bbox(mesh_path,1.0)

    mesh.vertices = vertices

    #sampling points on the surface of the bounded mesh (N, 3)
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, 8192//2)

    #point cloud & point normals
    point_cloud = torch.from_numpy(point_cloud).to(dtype=torch.float32)
    point_normals = torch.from_numpy(mesh.face_normals[face_indices]).to(dtype=torch.float32)

    # augmentation
    point_cloud = add_gaussian_noise(point_cloud, mean=0.0, std=0.01) #according to paper: mean = 0.0, std = 0.01
    point_normals = add_gaussian_noise(point_normals, mean=0.0, std=0.03)
    point_normals = set_zero_vector(points=point_normals, rate=0.3, size=point_normals.shape[1])

    points = torch.cat((point_cloud, point_normals), dim=1)

    return points

def test_inference():
    # Configure the model parameters based on the training model parameters

    generator = Inference(ConfigurationManager.model_params(), get_weights_path(ConfigurationManager.training_config(), 25))
    mesh_dir = ConfigurationManager.dataset_config().original_mesh_dir
    monkey_obj = os.path.join(mesh_dir, 'suzanne.obj')
    cube_obj = os.path.join(mesh_dir,'cube.obj')
    cone_obj = os.path.join(mesh_dir,'cone.obj')
    sphere_obj = os.path.join(mesh_dir,'sphere.obj')
    torus_obj = os.path.join(mesh_dir,'torus.obj')

    selected_obj = monkey_obj

    points = get_point_cloud_data(selected_obj).unsqueeze(0)
    face_count, quad_ratio = get_mesh_stats(selected_obj)
    face_count = torch.tensor(face_count, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    quad_ratio = torch.tensor(quad_ratio, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

    
    gen_point_cloud = generator.run(points, face_count, quad_ratio)

    write_obj(gen_point_cloud, os.path.join(get_root_folder(), 'artifacts','generations',f'gen_mesh_{os.path.basename(selected_obj)}'))


if __name__ == '__main__':
    main()
    # test_inference()