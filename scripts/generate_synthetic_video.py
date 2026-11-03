"""
Synthetic Video and Frame Generator for SUTRA Vision Engine.
Creates realistic mock experiment video frames simulating the 5-step biological assay,
including valid runs and intentional procedural deviation scenarios.
"""

import math
import os
import cv2
import numpy as np


def render_scene(frame_idx: int, total_frames: int = 180, simulate_deviation: bool = False) -> np.ndarray:
    """
    Render a single synthetic frame with animated hands, vials, pipettes, and fiducials.
    Resolution: 640 x 480 BGR.
    """
    w, h = 640, 480
    frame = np.full((h, w, 3), 45, dtype=np.uint8)  # Space station rack gray background

    # 1. Draw Payload Rack Boundary & 4 Corner Fiducials
    cv2.rectangle(frame, (40, 40), (w - 40, h - 40), (80, 80, 85), 2)
    fiducials = [(40, 40), (w - 40, 40), (w - 40, h - 40), (40, h - 40)]
    for fx, fy in fiducials:
        cv2.circle(frame, (fx, fy), 8, (0, 255, 255), -1)
        cv2.circle(frame, (fx, fy), 4, (0, 0, 0), -1)

    # 2. Draw Static Laboratory Elements
    # Waste Bin (bottom-left)
    cv2.rectangle(frame, (60, h - 140), (140, h - 50), (60, 60, 160), -1)
    cv2.putText(frame, "WASTE", (72, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Tip Box (top-left)
    cv2.rectangle(frame, (60, 60), (150, 130), (140, 100, 40), -1)
    cv2.putText(frame, "TIPS", (85, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Reagent / Sample Vial (center-right)
    # Yellow/Amber color so HeuristicLabDetector detects it as vial_sealed / vial_open
    vial_open = frame_idx >= 60
    vial_x, vial_y, vial_w, vial_h = 420, 260, 45, 80
    vial_color = (0, 165, 255)  # Amber in BGR
    cv2.rectangle(frame, (vial_x, vial_y), (vial_x + vial_w, vial_y + vial_h), vial_color, -1)
    if not vial_open:
        # Contrasting Cap
        cv2.rectangle(frame, (vial_x - 3, vial_y - 12), (vial_x + vial_w + 3, vial_y), (0, 0, 220), -1)
        cv2.putText(frame, "SEALED", (vial_x - 5, vial_y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "OPEN", (vial_x + 5, vial_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # 3. Dynamic Tool & Hand Trajectories based on Phase
    # Hand color: Skin tone (YCrCb range calibrated: e.g. BGR ~ [150, 180, 235])
    skin_color = (150, 180, 235)
    blue_tool = (230, 110, 30)  # Pipette in BGR

    hand_x, hand_y = 520, 420  # Rest position
    tool_x, tool_y = 260, 220  # Pipette cradle

    if frame_idx < 35:
        # Step 1: Inspection - Hand sweeps across workspace
        prog = frame_idx / 35.0
        hand_x = int(500 - prog * 240)
        hand_y = int(380 - math.sin(prog * math.pi) * 80)

    elif frame_idx < 75:
        # Step 2: Unsealing Vial - Hand reaches for vial and unseals it
        prog = (frame_idx - 35) / 40.0
        hand_x = int(260 + prog * (vial_x - 260))
        hand_y = int(300 + prog * (vial_y - 300))

    elif frame_idx < 115:
        # Step 3: Reagent Aspiration OR Skipped Step Deviation!
        if simulate_deviation:
            # DEVIATION: Hand ignores pipette aspiration and directly attempts dispensation or closes vial!
            prog = (frame_idx - 75) / 40.0
            hand_x = int(vial_x + 10)
            hand_y = int(vial_y + 10)
        else:
            # Normal: Hand grasps pipette at cradle and inserts tip into vial
            prog = (frame_idx - 75) / 40.0
            tool_x = int(260 + prog * (vial_x - 240))
            tool_y = int(220 + prog * (vial_y - 200))
            hand_x = tool_x + 20
            hand_y = tool_y + 15

    elif frame_idx < 150:
        # Step 4: Dispensation
        tool_x = vial_x - 5
        tool_y = vial_y - 10
        hand_x = tool_x + 20
        hand_y = tool_y + 15

    else:
        # Step 5: Container Sealing - Hand returns to vial cap, then exits
        prog = (frame_idx - 150) / 30.0
        hand_x = int(vial_x + prog * 100)
        hand_y = int(vial_y + prog * 120)
        tool_x, tool_y = 260, 220

    # Draw Pipette (elongated tool cylinder)
    if not (simulate_deviation and 75 <= frame_idx < 115):
        cv2.rectangle(frame, (tool_x, tool_y), (tool_x + 16, tool_y + 90), blue_tool, -1)
        cv2.circle(frame, (tool_x + 8, tool_y), 6, (255, 200, 0), -1)

    # Draw Hand (Palm circle + fingers)
    cv2.circle(frame, (hand_x, hand_y), 32, skin_color, -1)
    # Fingers
    cv2.circle(frame, (hand_x - 14, hand_y - 28), 9, skin_color, -1)
    cv2.circle(frame, (hand_x, hand_y - 34), 9, skin_color, -1)
    cv2.circle(frame, (hand_x + 14, hand_y - 28), 9, skin_color, -1)

    return frame


def create_synthetic_frame_generator(simulate_deviation: bool = False, total_frames: int = 180):
    """Returns a callable yielding sequential synthetic video frames."""
    idx = 0

    def generator():
        nonlocal idx
        f = render_scene(idx % total_frames, total_frames, simulate_deviation)
        idx += 1
        return f

    return generator


def generate_video_file(output_path: str, simulate_deviation: bool = False, total_frames: int = 180, fps: int = 15):
    """Render and write a synthetic MP4 video file to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    for i in range(total_frames):
        frame = render_scene(i, total_frames, simulate_deviation)
        out.write(frame)
    out.release()
    print(f"Generated synthetic video ({total_frames} frames, deviation={simulate_deviation}) -> {output_path}")


if __name__ == "__main__":
    generate_video_file("data/sample_normal_run.mp4", simulate_deviation=False)
    generate_video_file("data/sample_deviation_run.mp4", simulate_deviation=True)
