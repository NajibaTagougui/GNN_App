# ========================
# 1. Mount Google Drive
# ========================
from google.colab import drive
drive.mount('/content/drive',force_remount=True)

# ========================
# 2. Install PyDrive
# ========================
!pip install -U -q PyDrive2
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# ========================
# 3. COCO Dataset Setup in Drive
# ========================
import os
import zipfile
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as T
# Paths in Drive
DRIVE_ROOT = '/content/drive/MyDrive'
COCO_DIR = os.path.join(DRIVE_ROOT, 'coco_data')

# Create directory structure
os.makedirs(COCO_DIR, exist_ok=True)

def setup_coco_in_drive():
    # Download COCO directly to Drive
    !wget http://images.cocodataset.org/zips/val2017.zip -P {COCO_DIR}
    !wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {COCO_DIR}

    # Unzip
    with zipfile.ZipFile(os.path.join(COCO_DIR, 'val2017.zip'), 'r') as zip_ref:
        zip_ref.extractall(COCO_DIR)
    with zipfile.ZipFile(os.path.join(COCO_DIR, 'annotations_trainval2017.zip'), 'r') as zip_ref:
        zip_ref.extractall(COCO_DIR)

    print(f"COCO dataset ready at {COCO_DIR}")

# ========================
# 4. Modified Dataset Class for Drive
# ========================
class DriveCOCODataset(torch.utils.data.Dataset):
    def __init__(self, split='val2017', num_images=1000):
        self.img_dir = os.path.join(COCO_DIR, split)
        self.ann_file = os.path.join(COCO_DIR, 'annotations', f'instances_{split}.json')
        
        if not os.path.exists(self.img_dir) or not os.path.exists(self.ann_file):
            print("Dataset not found in Drive. Downloading...")
            setup_coco_in_drive()
            
        self.coco = COCO(self.ann_file)
        self.img_ids = self.coco.getImgIds()[:num_images]
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image directly from Drive
        img = Image.open(img_path).convert('RGB')
        img_tensor = T.ToTensor()(img)
        
        return img_tensor

# ========================
# 5. Full Workflow Example
# ========================
if __name__ == "__main__":
    # Initialize dataset (auto-downloads to Drive if needed)
    dataset = DriveCOCODataset(num_images=1000)
    
    # Example: Show first image
    import matplotlib.pyplot as plt
    plt.imshow(dataset[0].permute(1, 2, 0))
    plt.show()
