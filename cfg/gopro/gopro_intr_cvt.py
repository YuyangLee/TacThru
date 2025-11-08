import json


def scale_intrinsics(input_file, output_file, target_width, target_height):
    with open(input_file) as file:
        intrinsics_data = json.load(file)

    original_width = intrinsics_data["image_width"]
    original_height = intrinsics_data["image_height"]

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    intrinsics_data["image_width"] = target_width
    intrinsics_data["image_height"] = target_height

    intrinsics_data["intrinsics"]["focal_length"] *= scale_x
    intrinsics_data["intrinsics"]["principal_pt_x"] *= scale_x
    intrinsics_data["intrinsics"]["principal_pt_y"] *= scale_y

    with open(output_file, "w") as file:
        json.dump(intrinsics_data, file, indent=4)

    print(f"Intrinsics scaled and saved to {output_file}")


scale_intrinsics("gopro_intrinsics_2_7k.json", "gopro_intrinsics_720p.json", 960, 720)
