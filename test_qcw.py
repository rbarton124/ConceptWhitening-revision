import argparse
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch.nn.functional as F
from MODELS.ConceptDataset_QCW import ConceptDataset
from MODELS.model_resnet_qcw_bn import build_resnet_qcw
from masking_utils import mask_concepts, apply_per_layer_activation_masks, clear_all_activation_masks

def load_cw_resnet(checkpoint_path, concept_dir, depth=18, whitened_layers=[5],
                   act_mode="pool_max", vanilla_pretrain=False):
    """
    Loads CW ResNet model with trained weights.
    """
    concept_ds = ConceptDataset(
        root_dir=f"{concept_dir}/concept_train",
        bboxes_file=f"{concept_dir}/bboxes.json",
        high_level_filter=[], # pass in parameters from concepts
        transform=None,
        crop_mode="crop"
    )
    subspace_mapping = concept_ds.subspace_mapping

    model = build_resnet_qcw(
        num_classes=200,
        depth=depth,
        whitened_layers=whitened_layers,
        act_mode=act_mode,
        subspaces=subspace_mapping,
        use_subspace=True,
        vanilla_pretrain=vanilla_pretrain
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    return model, concept_ds

def evaluate_dataset(model, dataloader, concept_ds=None, concepts_to_mask=None):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    correct = 0
    total = 0
    total_loss = 0.0

    if concepts_to_mask and concept_ds:
        masks = mask_concepts(model, concept_ds, concepts_to_mask)
        apply_per_layer_activation_masks(model, masks)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    if concepts_to_mask and concept_ds:
        clear_all_activation_masks(model)

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss

def get_concept_loss(model):
    cw_layers = getattr(model.module, "cw_layers", [])
    if not cw_layers:
        print("No CW layers found")
        return 0.0

    valid_losses = []
    for layer in cw_layers:
        count = getattr(layer, "concept_loss_count", 0)
        if count > 0:
            loss = layer.get_concept_loss()
            valid_losses.append(loss)

    if not valid_losses:
        print(f"no valid losses")
        return 0.0
    
    print(f"valid losses: ", valid_losses)

    return sum(valid_losses) / len(valid_losses)

def collect_image_paths(class_folder_path):
    """
    Given a class folder path, returns a comma-separated string of all image file paths inside it.

    Example:
        Input: "/path/to/CUB/test/167"
        Output: "/path/to/CUB/test/167/img1.jpg,/path/to/CUB/test/167/img2.jpg,..."
    """
    if not os.path.isdir(class_folder_path):
        raise ValueError(f"Path '{class_folder_path}' is not a directory.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    all_files = os.listdir(class_folder_path)

    image_paths = [
        os.path.join(class_folder_path, f)
        for f in sorted(all_files)
        if os.path.splitext(f)[-1].lower() in image_extensions
    ]

    if not image_paths:
        raise ValueError(f"No image files found in '{class_folder_path}'.")

    return ",".join(image_paths)

def run_inference(model, image_paths, concepts_to_mask=None, concept_ds=None):
    """
    Runs inference on a list of image paths.
    Optionally mask specific concepts during inference.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # if concepts_to_mask and concept_ds:
    #     masks = mask_concepts(model, concept_ds, concepts_to_mask)
    #     apply_per_layer_activation_masks(model, masks)

    predictions = {}
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
            dummy_label = torch.tensor([pred]).cuda()
            loss = F.cross_entropy(output, dummy_label)

        predictions[img_path] = pred
        print(f"{img_path} --> Predicted Class: {pred}")
        # Concept Loss
        # cw_loss = get_concept_loss(model)

        print(f"  Classification Loss:   {loss.item():.4f}")
        # print(f"  Concept Loss:          {cw_loss:.4f}")

    if concepts_to_mask and concept_ds:
        clear_all_activation_masks(model)

    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--concept_dir", required=True)
    parser.add_argument("--data_test_dir", help="Optional: path to val/ or test/ set (ImageFolder structure).")
    parser.add_argument("--data_dir", required=True, help="Path to main dataset containing train/val/test subfolders (ImageFolder structure).")
    parser.add_argument("--image_paths", help="Optional: comma-separated image paths for individual testing.")
    parser.add_argument("--mask_concepts", default="", help="Concepts to mask")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--bboxes", default="", help="Path to bboxes.json if not in concept_dir/bboxes.json")
    parser.add_argument("--concepts", required=True, help="Comma-separated list of high-level concepts to use (e.g. 'wing,beak,general').")
    parser.add_argument("--prefix", required=True, help="Prefix for logging & checkpoint saving")
    parser.add_argument("--batches_per_concept", type=int, default=1, help="Number of batches per subconcept for each alignment step.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--cw_lambda", type=float, default=0.1, help="Lambda parameter for QCW.")
    parser.add_argument("--whitened_layers", default="5")
    parser.add_argument("--act_mode", default="pool_max")
    parser.add_argument("--vanilla_pretrain", action="store_true")

    args = parser.parse_args()

    whitened_layers = [int(x) for x in args.whitened_layers.split(",") if x.strip() != ""]
    concepts = [x.strip() for x in args.mask_concepts.split(",") if x.strip() != ""]

    model, _ = load_cw_resnet(
        checkpoint_path=args.checkpoint,
        concept_dir=args.concept_dir,
        depth=args.depth,
        whitened_layers=whitened_layers,
        act_mode=args.act_mode,
        vanilla_pretrain=args.vanilla_pretrain
    )

    # Build subconcept loaders
    concept_loaders, concept_ds = build_concept_loaders(args)
    subconcept_loaders = concept_loaders[1:]

    # Apply concept masking BEFORE alignment
    if concepts and concept_ds:
        print(f"Applying masking for alignment: {concepts}")
        masks = mask_concepts(model, concept_ds, concepts)
        apply_per_layer_activation_masks(model, masks)
        print(f"Masked indices:", [i for i, m in enumerate(masks[0]) if m == 0.0])

    # Run alignment for analysis (this updates internal CW loss counters)
    align_concepts(
        model,
        subconcept_loaders,
        concept_ds,
        batches_per_concept=args.batches_per_concept,
        lambda_=args.cw_lambda
    )

    concept_loss = get_concept_loss(model)
    print(f"  Concept Loss:          {concept_loss:.4f}")

    if args.data_test_dir:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(args.data_test_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        acc, loss = evaluate_dataset(model, dataloader, concept_ds=concept_ds, concepts_to_mask=concepts if concepts else None)
        cw_loss = get_concept_loss(model)

        print(f"Dataset: {args.data_dir}")
        print(f"Top-1 Accuracy: {acc:.2f}%")
        print(f"Avg Classification Loss: {loss:.4f}")
        print(f"Avg Concept Loss: {cw_loss:.4f}")

    elif args.image_paths:
        print(f"running inference on specified image path...")
        # image_list = [x.strip() for x in args.image_paths.split(",") if x.strip() != ""]
        image_list = collect_image_paths(args.image_paths).split(",")
        run_inference(model, image_list, concepts_to_mask=concepts if concepts else None, concept_ds=concept_ds)
        print(f"masked concepts: ", concepts)

    else:
        print("Provide either --data_dir or --image_paths to run inference.")

if __name__ == "__main__":
    main()
