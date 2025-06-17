import torch
import torchvision
import torchvision.transforms as transforms
from train_model import SimpleCNN

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

model = SimpleCNN()
model.load_state_dict(torch.load("model/cnn_model.pt"))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Model Accuracy: {accuracy:.2f}%")

if accuracy < 40:
    raise Exception("Model accuracy below threshold!")
