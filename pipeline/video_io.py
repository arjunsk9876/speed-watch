import av
import cv2


def iter_frames(video_path):
    """Stream a video one BGR frame at a time (does not load the whole clip into memory)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def sample_frames(video_path, n=6):
    """Evenly sample n frames (as BGR numpy arrays) from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


class VideoWriter:
    """Incremental H.264 mp4 writer so annotated frames can be written as they're produced.

    Uses PyAV/libx264 rather than cv2's default `mp4v` fourcc, which produces
    files roughly 5x larger for the same content (e.g. 120MB vs 25MB for a
    23-minute 640x360 clip) and can exceed GitHub's 100MB file limit.
    """

    def __init__(self, output_path, fps, frame_size, crf=30):
        width, height = frame_size
        self._container = av.open(str(output_path), mode="w")
        self._stream = self._container.add_stream("h264", rate=int(round(fps)))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(crf), "preset": "medium"}

    def write(self, frame):
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def release(self):
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def get_fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_frame_size(video_path):
    """Returns (width, height)."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height
