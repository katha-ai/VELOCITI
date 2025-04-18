import cv2
import concurrent.futures
from os.path import join
from glob import glob
from tqdm import tqdm
from pathlib import Path
import argparse


def extract_frames(vid, frames_dir):
    vid_name = vid.split('/')[-1].replace('.mp4', '')
    cap = cv2.VideoCapture(vid)

    if (cap.isOpened()== False): 
        print("Error opening video file", vid.split('/')[-1])
        return

    folder_name = join(frames_dir, vid_name)
    Path(join(folder_name)).mkdir(parents=True, exist_ok=True)

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    if fps != 24:
        print("Vid name =", vid_name, "FPS =", fps)    	

    i = 1
    frame_nums = []
    interval_list = [0.25, 0.75]
    sampling_rate = len(interval_list)

    for second in range(10):
        for interval in interval_list:
            frame_nums.append(round((interval + second) * fps))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            if i in frame_nums:
                cv2.imwrite(join(folder_name, 'frame_' + str(i).zfill(3) + '.jpg'), frame) 
            i += 1
        
        else:
            break

    return "Done sampling... " + vid_name


def main(args):

    vid_list = glob(join(args.videos_dir, '*.mp4'))

    with concurrent.futures.ThreadPoolExecutor(10) as executor:

        results = [executor.submit(extract_frames, vid_list[i], args.frames_dir) for i in range(len(vid_list))]
        
        _ = [r.result() for r in tqdm(concurrent.futures.as_completed(results), total = len(results))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--videos_dir",
        default="./videos",
        type=str,
        help="Directory to where downloaded videos are stored",
    )
    parser.add_argument(
        "--frames_dir",
        default="./frames",
        type=str,
        help="Directory to where frames would be stored",
    )

    args = parser.parse_args()
    main(args)