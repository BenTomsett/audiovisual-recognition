import time
import matlab.engine
from pathlib import Path

print("Starting MATLAB Engine...", end="")
eng = matlab.engine.start_matlab()
print(" done.")

eng.cd(r"video_features/no_blocks", nargout=0)

folder_path = Path("./videos_cropped").absolute().as_posix()

start_time = time.time()

eng.computeAllDctCoefficients(folder_path, nargout=0)

end_time = time.time()

print("Completed in ", end_time - start_time, " seconds")
