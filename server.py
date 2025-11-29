from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
from pipeline.stages.inference import get_generator
from pipeline.utils.common import get_path, get_root_folder
import uvicorn

generator = get_generator(get_path(get_root_folder(), 'artifacts', 'meshtron_75.pt'))

app = FastAPI()

async def generate_tokens(points, face_count, quad_ratio):
    for coord in generator.run(points, face_count, quad_ratio):
        yield f"{coord}\n"
        await asyncio.sleep(0)

@app.post("/stream")
async def stream(data: dict):
    points = data["point_cloud"]
    face_count = data["face_count"]
    quad_ratio = data["quad_ratio"]

    return StreamingResponse(
        generate_tokens(points, face_count, quad_ratio),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
