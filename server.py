from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
from pipeline.stages.inference import get_generator
from pipeline.utils.common import get_path, get_root_folder
import uvicorn
import torch

generator = get_generator(get_path(get_root_folder(), 'artifacts', 'models', 'meshtron_75.pt'))

app = FastAPI()

async def generate_tokens(points, face_count, quad_ratio):
    for coord in generator.run(points, face_count, quad_ratio):
        yield f"{coord}\n"
        await asyncio.sleep(0)

@app.post("/stream")
async def stream(data: dict):
    points = torch.tensor(data["point_cloud"])
    face_count = torch.tensor(data["face_count"])
    quad_ratio = torch.tensor(data["quad_ratio"])

    return StreamingResponse(
        generate_tokens(points, face_count, quad_ratio),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
