import argparse
import os
import imageio
import pyrender
import trimesh
import numpy as np
from tqdm import tqdm


def look_at(eye, target=[0, 0, 0], up=[0, 1, 0]):
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = (target - eye)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    if np.linalg.norm(right) == 0:
        right = np.array([1, 0, 0], dtype=np.float64)
    else:
        right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def render_glb_to_video(glb_path, output_video_path, n_frames=120, radius=2.5):
    mesh_or_scene = trimesh.load(glb_path)

    if isinstance(mesh_or_scene, trimesh.Scene):
        combined = trimesh.util.concatenate(list(mesh_or_scene.geometry.values()))
        mesh = combined
    else:
        mesh = mesh_or_scene

    scene = pyrender.Scene()

    center = mesh.centroid if hasattr(mesh, 'centroid') else np.array([0, 0, 0])

    # ✅ SpotLight 위에서 아래로 추가
    spotlight = pyrender.SpotLight(
        color=np.ones(3),
        intensity=8.0,
        innerConeAngle=np.pi / 10,
        outerConeAngle=np.pi / 6
    )
    spotlight_pose = look_at([0, 5, 0], target=center)
    scene.add(spotlight, pose=spotlight_pose)

    # 전방위 DirectionalLight 보조로 유지 (원한다면 제거 가능)
    light_dirs = [
        [ 2, 2, 2], [-2, 2, 2], [2, 2, -2], [-2, 2, -2]
    ]
    for dir in light_dirs:
        pose = look_at(dir, target=center)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=pose)

    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = look_at([radius, 0.0, 0.0], target=center)
    cam_node = scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
    frames = []

    for i in tqdm(range(n_frames), desc=f"Rendering {os.path.basename(glb_path)}"):
        angle = 2 * np.pi * i / n_frames
        cam_x = radius * np.cos(angle)
        cam_z = radius * np.sin(angle)
        cam_y = 1.0

        camera_pose = look_at([cam_x, cam_y, cam_z], target=center)
        scene.set_pose(cam_node, pose=camera_pose)

        color, _ = renderer.render(scene)
        frames.append(color)

    renderer.delete()
    imageio.mimsave(output_video_path, frames, fps=25)
    print(f"✅ Saved: {output_video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input .glb file")
    parser.add_argument("--output", "-o", required=True, help="Output .mp4 path")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--radius", type=float, default=3, help="Camera orbit radius")
    args = parser.parse_args()

    render_glb_to_video(
        glb_path=args.input,
        output_video_path=args.output,
        n_frames=args.frames,
        radius=args.radius
    )


if __name__ == "__main__":
    main()
