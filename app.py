import numpy as np
import torch
import os
from PIL import Image
import streamlit as st


def main():
    st.title("Recommendation system")

    # Ввод артикула
    article_main = st.text_input("Введите артикул:")

    # Отображение изображений
    if article_main:
        articles = np.load(r"D:\Диплом_data\articles.npy")
        folder_neighbours = r"D:\Диплом_data\neighbours_cat"
        folder_images = r"D:\Диплом_data\resized_images_128x128"
        neighbours = []
        for article in articles:
            with torch.no_grad():
                neighbours.append(torch.load(f'{folder_neighbours}\{article}.pt', map_location='cpu'))

        neighbours = torch.stack(neighbours).numpy()
        try:
            paths = neighbours[list(articles).index(int(article_main))][:6]
            paths = [os.path.join(folder_images, str(path) + ".jpg") for path in paths]

            # Проверка наличия путей к изображениям
            if len(paths) == 6:
                # Отображение главного изображения слева
                main_image = Image.open(paths[0])
                st.image(main_image, caption='Исходная сережка', use_column_width=False)

                # Отображение 5 центральных изображений
                col1, col2, col3, col4, col5 = st.columns(5)
                for path, col in zip(paths[1:], [col1, col2, col3, col4, col5]):
                    jewellery_image = Image.open(path)
                    article = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
                    domain = "https://sunlight.net/catalog/earring_"
                    link = domain + article + ".html"
                    col.image(jewellery_image, caption=link, use_column_width=True)

            else:
                st.warning("Не удалось найти все изображения для заданного артикула.")
        except:
            st.warning("Данный артикул не найден")


if __name__ == "__main__":
    main()