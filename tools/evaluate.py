import torch 
import tqdm

def evaluate(model, dataloader, device, logger):
    # again no gradients needed
    model.eval()

    import numpy as np
    from prettytable import PrettyTable

    if hasattr(dataloader.dataset, 'datasets'):
        # if dataset is concat, bring the num of classes of the first
        classes = dataloader.dataset.datasets[0].classes 
        num_classes = len(classes)
    else: 
        classes = dataloader.dataset.classes
        num_classes = len(classes)
    
    with torch.no_grad():
        category_vectors = []
        for data in tqdm(dataloader, desc='Evaluation:'):
            inputs, labels = data['image'], data['segmap']

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            category_vector = torch.flatten(predictions * num_classes + labels)
            category_vector = category_vector.detach().cpu().numpy()

            category_vectors.append(category_vector)

        category_vectors = np.concatenate(category_vectors)
        category_bincount = np.bincount(category_vectors, minlength = num_classes ** 2)

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        for i in range(num_classes ** 2):
            row, col = i // num_classes, i % num_classes
            confusion_matrix[row, col] = category_bincount[i] 

        ious = []
        for i in range(num_classes): 
            intersection = confusion_matrix[i, i]
            union = np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
            ious.append(intersection / union)
        
        miou = np.mean(ious)

        iou_table = PrettyTable(['Class', 'IoU (%)'])
        
        for idx, iou in enumerate(ious):
            iou_table.add_row([classes[idx], f'{iou*100:.2f}'])

        miou_table = PrettyTable(['Metric', '(%)' ])
        miou_table.add_row(['mIoU', f'{miou*100:.2f}'])

        logger.info('\n'+iou_table.get_string())
        logger.info('\n'+miou_table.get_string())
        
            
    model.train()
    return miou, ious
