import os
import shutil
import random
from PIL import Image
from pathlib import Path

def re_or_create_directory(dir_path: Path):
    """Очищает директорию по пути или создает новую"""
    if dir_path.is_dir():
        for item in dir_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        dir_path.mkdir(parents=True)

def parse_classes(input_path: Path) -> list[str]:
    """Парсит классификаторы"""
    classes: list[str] = []
    file_path = input_path / 'classes.txt'
    if not file_path.is_file():
        raise FileNotFoundError(f'classes.txt not found in {input_path}')
    with file_path.open('r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            classes.append(line.strip())
    if not classes:
        raise ValueError('No classes were parsed')
    return classes

def create_directory_structure(
        output_dir: Path, 
        class_names: list[str], 
    ):
    """Создает структуру директорий для YOLOv8."""
    dirs = ['train', 'valid', 'test']

    re_or_create_directory(output_dir)
    
    data_yaml = output_dir / 'data.yaml'

    with data_yaml.open('w') as yaml_file:
        for split in dirs:
            if split == 'valid':
                yaml_file.write(f'val: {os.path.join("..", "valid", "images")}\n')
            else:
                yaml_file.write(f'{split}: {os.path.join("..", split, "images")}\n')
        yaml_file.write(f'nc: {len(class_names)}\n')
        yaml_file.write(f'names: {class_names}\n')

    for split in dirs:
        output_image_dir = (output_dir / split) / 'images'
        output_label_dir = (output_dir / split) / 'labels'
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

from PIL import Image
from pathlib import Path

def resize_images_labels(source_dir: Path, output_dir: Path, size: tuple[int, int] = (640, 640)):
    temp_dir = output_dir / 'temp'
    temp_images = temp_dir / 'images'
    temp_labels = temp_dir / 'labels'

    temp_images.mkdir(parents=True, exist_ok=True)
    temp_labels.mkdir(parents=True, exist_ok=True)

    source_images_dir = source_dir / 'images'
    source_labels_dir = source_dir / 'labels'

    t_width, t_height = size

    for image_file in source_images_dir.glob('*.png'):
        # Изменение размера изображения
        with Image.open(image_file) as img:
            og_width, og_height = img.size

            # Масштабируем изображение, если оно не соответствует целевому размеру
            if og_width != t_width or og_height != t_height:
                scale_x = t_width / og_width
                scale_y = t_height / og_height

                img = img.resize(size, Image.Resampling.BILINEAR)
                img.save(temp_images / image_file.name)

                label_file = source_labels_dir / (image_file.stem + '.txt')
                if label_file.is_file():
                    with open(label_file, 'r') as file:
                        lines = file.readlines()
                        corrected_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id, x_center, y_center, width, height = map(float, parts)
                                # Переводим в абсолютные координаты, масштабируем и обратно в относительные
                                x_center = (x_center * og_width) * scale_x / t_width
                                y_center = (y_center * og_height) * scale_y / t_height
                                width = (width * og_width) * scale_x / t_width
                                height = (height * og_height) * scale_y / t_height

                                corrected_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
                                corrected_lines.append(corrected_line)
                        with open(temp_labels / label_file.name, 'w') as file:
                            file.writelines(corrected_lines)
            else:
                # Если изображение уже нужного размера, просто копируем без изменений
                img.save(temp_images / image_file.name)
                label_file = source_labels_dir / (image_file.stem + '.txt')
                if label_file.is_file():
                    with open(label_file, 'r') as file:
                        with open(temp_labels / label_file.name, 'w') as output_file:
                            output_file.writelines(file.readlines())


def distribute_dataset(
        output_dir: Path,
        train_size: float = 0.8,
        valid_size: float = 0.1
    ):
    assert 0 < train_size < 1, "train_size must be a float between 0 and 1"
    assert 0 <= valid_size < 1, "valid_size must be a float between 0 and 1"
    assert train_size + valid_size <= 1, "train_size + valid_size must be less or equal to 1"

    images_temp = (output_dir / 'temp') / 'images'
    labels_temp = (output_dir / 'temp') / 'labels'

    train_images = output_dir / 'train/images'
    train_labels = output_dir / 'train/labels'
    valid_images = output_dir / 'valid/images'
    valid_labels = output_dir / 'valid/labels'
    test_images = output_dir / 'test/images'
    test_labels = output_dir / 'test/labels'

    # Получаем список файлов изображений
    image_files = list(images_temp.glob('*.png'))
    random.shuffle(image_files)  # Перемешиваем файлы

    train_end = int(len(image_files) * train_size)
    valid_end = train_end + int(len(image_files) * valid_size)

    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]

    # Функция для копирования файлов
    def copy_files(files, img_dest, lbl_dest):
        for file in files:
            shutil.copy2(file, img_dest)
            label_file = labels_temp / (file.stem + '.txt')
            shutil.copy2(label_file, lbl_dest)

    # Копирование файлов
    copy_files(train_files, train_images, train_labels)
    copy_files(valid_files, valid_images, valid_labels)
    copy_files(test_files, test_images, test_labels)

    print(f"Dataset distribution completed:\n"
          f"Training set: {len(train_files)} images\n"
          f"Validation set: {len(valid_files)} images\n"
          f"Test set: {len(test_files)} images")

def delete_temp(output_dir: Path):
    temp_dir = output_dir / 'temp'
    if temp_dir.is_dir():
        shutil.rmtree(temp_dir)

def main():
    
    cur_dir = Path.cwd()
    source_dir = cur_dir / 'source'
    class_names = parse_classes(source_dir)
    output_dir = cur_dir / 'output'
    create_directory_structure(output_dir, class_names)
    resize_images_labels(source_dir, output_dir)
    distribute_dataset(output_dir)
    delete_temp(output_dir)

if __name__ == '__main__':
    main()