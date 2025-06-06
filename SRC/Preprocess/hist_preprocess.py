import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import pandas as pd
from tqdm import tqdm

def histology_preprocess(image_dir: str,
                         manifest: pd.DataFrame,
                         tile_size: int = 224,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Preprocess PNG histology images by extracting a center tile, passing through ResNet, and returning feature vectors.

    :param image_dir: Directory containing .png images.
    :param tile_size: Size of the center tile to extract.
    :param device: 'cuda' or 'cpu'
    :return: DataFrame with file_id and ResNet features as rows.
    """

    # Load pretrained ResNet and remove final classification layer
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
    model.to(device)
    model.eval()

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    feature_list = []
    file_ids = []

    # Loop through PNG files in directory
    for filename in tqdm(os.listdir(image_dir), desc=f'Processing histology images by resnet encode of image centre crop {tile_size} pixels'):
        if filename.lower().endswith('.png'):
            file_id = os.path.splitext(filename)[0]
            file_id = file_id.split('.')[0].split('-')[:4]
            #join with '-'
            file_id = '-'.join(file_id)
            #replace end Z with A lol
            file_id = file_id[:-1] + 'A'


            image_path = os.path.join(image_dir, filename)

            try:
                image = Image.open(image_path).convert('RGB')

                # Extract center crop
                width, height = image.size
                left = (width - tile_size) // 2
                top = (height - tile_size) // 2
                right = left + tile_size
                bottom = top + tile_size
                image_cropped = image.crop((left, top, right, bottom))

                input_tensor = transform(image_cropped).unsqueeze(0).to(device)

                with torch.no_grad():
                    features = model(input_tensor).squeeze().cpu().numpy()  # shape (2048,)

                file_ids.append(file_id)
                feature_list.append(features)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # Build DataFrame
    feature_df = pd.DataFrame(feature_list, index=file_ids)
    feature_df.index.name = 'file_id'
    feature_df.reset_index(inplace=True)


    final_df = feature_df.merge(manifest, left_on='file_id', right_on='submitter_id.samples', how='inner')

    train_df = final_df[final_df['split'] == 'train'].drop(columns=['file_id','split'])
    val_df = final_df[final_df['split'] == 'val'].drop(columns=['file_id','split'])
    test_df = final_df[final_df['split'] == 'test'].drop(columns=['file_id','split'])

    return train_df, val_df, test_df


def wrangle_disgusting_file_ids(list) -> list:

    """
    Cleans up file IDs by removing unwanted characters and formatting them correctly.
    :param list: List of file IDs to clean.
    :return: Cleaned list of file IDs.
    """
    cleaned_list = []
    for item in list:
        item = item.split('.')[0]  # Remove file extension
        item = item.split('-')[:4]  # Keep only the first four parts
        item = '-'.join(item)  # Join back with '-'
        item = item[:-1] + 'A'  # Replace last character with 'A'
        cleaned_list.append(item)
    return cleaned_list

def foundation_histology_preprocess(input_df, manifest):

    file_ids = input_df['file_name'].tolist()

    cleaned_file_ids = wrangle_disgusting_file_ids(file_ids)

    input_df['file_id'] = cleaned_file_ids
    input_df.drop(['file_name'], axis=1, inplace=True)

    # Merge with manifest to get the split and subtype information
    merged_df = input_df.merge(manifest, left_on='file_id', right_on='submitter_id.samples', how='inner')

    train_df = merged_df[merged_df['split'] == 'train'].drop(columns=['file_id','split'])
    val_df = merged_df[merged_df['split'] == 'val'].drop(columns=['file_id','split'])
    test_df = merged_df[merged_df['split'] == 'test'].drop(columns=['file_id','split'])

    return train_df, val_df, test_df





if __name__ == "__main__":
    # #image_directory = r"C:\Projects\Notebook_sandbox\hackathon\wsi\pngs_and_masks"
    # image_directory = r"C:\Projects\Notebook_sandbox\hackathon\wsi\test_dir"
    # manifest = pd.read_csv("../../data/hackathon_manifest.csv")
    # tile_size = 1024  # Size of the center tile to extract
    #
    # train, val, test = histology_preprocess(image_directory, manifest, tile_size)
    #
    # #print the shapes
    # print("Train shape:", train.shape)
    # print("Validation shape:", val.shape)
    # print("Test shape:", test.shape)
    #
    raw_data = pd.read_csv("../../data/he.csv")
    manifest = pd.read_csv("../../data/hackathon_manifest.csv")

    train_df, val_df, test_df = foundation_histology_preprocess(raw_data, manifest)
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)