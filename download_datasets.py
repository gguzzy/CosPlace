import os
import gdown
import shutil
import map_builder
from os.path import join, basename, splitext

download_urls = {
    #"tokyo_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link",
    #"sf_xs": "https://drive.google.com/file/d/1tQqEyt3go3vMh4fj_LZrRcahoTbzzH-y/view?usp=share_link",
    # "gsv_xs": "https://drive.google.com/file/d/15QB3VNKj93027UAQWv7pzFQO1JDCdZj2/view?usp=share_link" #not used
    "st_lucia_database": "https://mega.nz/file/nE4g0LzZ#c8eL_H3ZfXElqEukw38i32p5cjwusTuNJYYeEP1d5Pg", #st lucia db pass
    "st_lucia_queries":"https://mega.nz/file/PAgWSIhD#UeeA6knWL3pDh_IczbYkcA1R1MwSZ2vhEg2DTr1_oNw",
}


os.makedirs("data", exist_ok=True)
for dataset_name, url in download_urls.items():
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    #gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data") #unpacking zip (unzipping)
    os.remove(zip_filepath)

# extract information from the downloaded datasets (st.lucia)
datasets_folder = join(os.curdir, "data")

for dataset_name, url in download_urls.items():
    zip_filepath = f"data/{dataset_name}"
    if dataset_name.startswith("st_lucia"):
        # insert the code to extract information
        vr = util.VideoReader(join(subset_folder, "webcam_video.avi"))

        with open(join(subset_folder, "fGPS.txt"), "r") as file:
            lines = file.readlines()

        last_coordinates = None
        for frame_num, line in zip(tqdm(range(vr.frames_num)), lines):
            latitude, longitude = line.split(",")
            latitude = "-" + latitude  # Given latitude is positive, real latitude is negative (in Australia)
            easting, northing = util.format_location_info(latitude, longitude)[:2]
            if last_coordinates is None:
                last_coordinates = (easting, northing)
            else:
                distance_in_meters = util.get_distance(last_coordinates, (easting, northing))
                if distance_in_meters < THRESHOLD_IN_METERS:
                    continue  # If this frame is too close to the previous one, skip it
                else:
                    last_coordinates = (easting, northing)

            frame = vr.get_frame_at_frame_num(frame_num)
            image_name = util.get_dst_image_name(latitude, longitude, pano_id=f"{subset_name}_{frame_num:05d}")
            if sequence_num == 0:  # The first sequence is the database
                io.imsave(join(dst_database_folder, image_name), frame)
            else:
                io.imsave(join(dst_queries_folder, image_name), frame)

            sequence_num += 1

            map_builder.build_map_from_dataset(dataset_folder)
            shutil.rmtree(raw_data_folder)

    pass
    # done with operations
    # create tree (map_builder)
# done
