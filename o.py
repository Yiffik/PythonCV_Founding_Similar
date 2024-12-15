import cv2
import numpy as np
from imgaug import augmenters as iaa

# Глобальные переменные для управления масштабированием
scale_factor = 1.0
max_scale = 3.0
min_scale = 0.1

# Функция для применения фильтра Собеля
def apply_sobel_filter(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение фильтра Собеля по x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Применение фильтра Собеля по y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Комбинирование результатов фильтрации по x и y
    sobel_combined = cv2.magnitude(sobelx, sobely)
    # Нормализация результатов для отображения
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return sobel_combined

# Функция для поиска совпадений между шаблоном и изображением
def find_similar_regions(image, template, threshold=0.5):
    if image is None or template is None:
        return []

    # Применение метода сопоставления  шаблонов
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = []
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        locations.append(pt)

    return locations

# Функция для масштабирования изображения
def scale_image(image, scale):
    h, w = image.shape[:2]
    new_h = int(h * scale)
    new_w = int(w * scale)
    scaled_image = cv2.resize(image, (new_w, new_h))
    return scaled_image

# Функция для обработки событий  мыши
def mouse_callback(event, x, y, flags, param):
    global scale_factor

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            # Колесо мыши вверх - увеличение  масштаба
            scale_factor = min(scale_factor + 0.1, max_scale)
        else:
            # Колесо мыши вниз - уменьшение масштаба
            scale_factor = max(scale_factor - 0.1, min_scale)
        
        # Обновление отображаемого  изображения с новым масштабом
        scaled_image = scale_image(image_fullscreen, scale_factor)
        cv2.imshow("Fullscreen Image", scaled_image)

# Функция для ввода текста
def get_text_input():
    text = input("Введите текст для отображения внутри прямоугольников: ")
    return text

# Функция для проверки, пересекаются ли два прямоугольника
def is_inside(outer, inner):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih

# Путь к изображению
image_path = "/home/yiffik/present/5.png"

# Чтение изображения в цветном формате
image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image_color is None:
    print(f"Ошибка: Не удалось прочитать изображение из {image_path}")
    exit()

# Создание копии для отображения в полноэкранном режиме
image_fullscreen = image_color.copy()

# Применение фильтра Собеля к исходному изображению
sobel_filtered_image = apply_sobel_filter(image_color)

# Использование OpenCV для выбора областей интереса (ROI)
image_copy = image_color.copy()
rois = []
texts = []  # Список для хранения текста для каждого ROI
win_name = "Select ROI"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(win_name, lambda *args: None)  # Заглушка для обработки событий мыши
cv2.imshow(win_name, image_copy)
cv2.moveWindow(win_name, 0, 0)  # Перемещаем окно в левый верхний угол

while True:
    # Выбор области интереса
    r = cv2.selectROI(win_name, image_copy, fromCenter=False)
    if r == (0, 0, 0, 0):
        break

    # Проверка на вложенность
    is_nested = any(is_inside(existing_roi, r) for existing_roi in rois)

    if not is_nested:
        rois.append(r)
        text = get_text_input()  # Ввод текста для текущего ROI
        texts.append(text)
        # Рисование синих прямоугольников
        cv2.rectangle(image_copy, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 2)
        cv2.imshow(win_name, image_copy)

cv2.destroyAllWindows()

# Проверка, что области интереса были выбраны
if rois:
    # Копия исходного изображения для отображения результатов
    image_with_rectangles = image_color.copy()

    # Открываем файл для записи информации о прямоугольниках
    with open("/home/yiffik/present/result_coordinates.txt", "w") as file:
        rectangle_number = 1
        # Рисование прямоугольников вокруг найденных регионов для каждой области интереса
        for idx, roi in enumerate(rois):
            x, y, w, h = roi
            roi_image = image_color[y:y+h, x:x+w]

            # Поиск схожих элементов в обработанном изображении до аугментации
            locations = find_similar_regions(sobel_filtered_image, apply_sobel_filter(roi_image))

            # Аугментация области интереса с помощью imgaug
            aug = iaa.Sequential([
                iaa.Rot90((0, 3)),  # Случайный поворот на 0, 90, 180 или 270 градусов
                iaa.AdditiveGaussianNoise(scale=(0, 30)),  # Добавление гауссовского шума
            ])

            augmented_images = aug(images=[roi_image])
            augmented_roi = augmented_images[0]

            # Применение фильтра Собеля к аугментированной области
            sobel_filtered_roi = apply_sobel_filter(augmented_roi)
            
            # Поиск схожих элементов в обработанном изображении после аугментации
            locations_augmented = find_similar_regions(sobel_filtered_image, sobel_filtered_roi)

            # Рисование прямоугольников вокруг найденных регионов
            for loc in locations + locations_augmented:
                top_left = loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(image_with_rectangles, top_left, bottom_right, (0, 0, 255), 2)
                # Отображение текста внутри прямоугольников
                cv2.putText(image_with_rectangles, texts[idx], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Запись информации о прямоугольнике в файл
                file.write(f"{rectangle_number}, {texts[idx]}, ({top_left[0]}, {top_left[1]}), ({bottom_right[0]}, {bottom_right[1]})\n")
                rectangle_number += 1

    # Сохранение итогового изображения с обведенными прямоугольниками и текстом
    result_image_path = "/home/yiffik/present/result.png"
    cv2.imwrite(result_image_path, image_with_rectangles)

    print(f"Сохранено по пути: {result_image_path}")
else:
    print("Области интереса не были выбраны.")
