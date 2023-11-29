from PIL import Image
import os
from torchvision import transforms, datasets

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
])

# Define a function to apply augmentations and save images
def augment_and_save_images(dataset, augmentations, output_dir_base='path_to_augmented_data'):
    for idx, (image, label) in enumerate(dataset):
        class_name = dataset.classes[label]
        output_dir = os.path.join(output_dir_base, class_name)
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        image = transforms.ToPILImage()(image)
        
        # Apply each augmentation 5 times per image
        for aug_idx in range(5):
            augmented_image = augmentations(image)

            # Save the augmented image
            output_path = os.path.join(output_dir, f"{idx}_{aug_idx}.png")
            augmented_image.save(output_path, 'PNG')

path_to_data = r'C:\School\csci 4353\studentData'
dataset = datasets.ImageFolder(root=path_to_data, transform=transforms.ToTensor())
# Run the function
augment_and_save_images(dataset, augmentations, path_to_data)
