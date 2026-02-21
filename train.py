import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

# 导入你写好的模型结构
from models_def import ResNet50WithAttention

def main():
    # ================= 1. 配置参数 =================
    dataset_path = "dataset"  # 存放图片的文件夹名称
    save_dir = "applied_best_models" # 模型保存的文件夹
    model_save_path = os.path.join(save_dir, "best_model_resnet50.pth")
    
    batch_size = 8       # 每次喂给模型几张图，如果电脑显存大可以调成 16 或 32
    num_epochs = 10      # 训练轮数（测试阶段10轮即可，正式训练建议30-50轮）
    learning_rate = 0.001
    
    # 确保保存模型的文件夹存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 检测是否有显卡加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # ================= 2. 数据处理与加载 =================
    # 这里的参数必须和你 model_manager.py 里的保持完全一致！
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(dataset_path):
        print(f"错误：找不到数据集文件夹 '{dataset_path}'，请按照提示建立文件夹并放入图片！")
        return

    # 加载数据集
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(f"成功加载数据集，发现类别: {full_dataset.classes}")
    
    # 注意：ImageFolder 默认按字母顺序排序类别。
    # Faulty(0), Healthy(1)。为了配合你代码里的逻辑，最好确保对应。
    
    # 将数据集按 8:2 划分为训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"总图片数: {len(full_dataset)} | 训练集: {train_size} | 验证集: {val_size}")

    # ================= 3. 初始化模型、损失函数和优化器 =================
    model = ResNet50WithAttention(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_accuracy = 0.0

    # ================= 4. 开始训练 =================
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train

        # ================= 5. 验证模型 =================
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # ================= 6. 保存最佳模型 =================
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"⭐ 发现更好模型，已保存至: {model_save_path}")

    print("训练结束！你现在可以运行 main.py 来测试你的界面和模型了！")

if __name__ == "__main__":
    main()