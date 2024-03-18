import csv
import datetime
import os
import random
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_urls(url):
    # Funkcja pobierająca adresy URL z podanej strony internetowej
    response = requests.get(url)
    html_code = response.text

    soup = BeautifulSoup(html_code, 'html.parser')

    elements = soup.select('.ooa-e0hsj.erf161l2')
    urls = [url.find('a', class_="erf161l0 ooa-1dzmiqs er34gjf0")['href'] for url in
            elements[1].select('.ooa-1xclg2i.erf161l1')]
    return urls


def get_sub_urls(url):
    # Funkcja pobierająca podadresy URL z podanej strony internetowej
    response = requests.get(url)
    html_code = response.text

    soup = BeautifulSoup(html_code, 'html.parser')

    elements = soup.select('.ooa-e0hsj.erf161l2')
    urls = [url.find('a', class_="erf161l0 ooa-1dzmiqs er34gjf0")['href'] for url in
            elements[0].select('.ooa-1xclg2i.erf161l1')]
    return urls


def get_links():
    # Funkcja pobierająca wszystkie linki do ofert samochodów
    main_url = "https://www.otomoto.pl/osobowe"
    urls = get_urls(main_url)
    links = []
    for link in urls:
        sub_urls = get_sub_urls(link)
        for sub_link in sub_urls:
            links.append(sub_link)
    return links


def get_number_of_pages(url):
    # Funkcja pobierająca liczbę stron dla danego linku
    response = requests.get(url)
    html_code = response.text

    soup = BeautifulSoup(html_code, 'html.parser')

    element = soup.find(class_='ooa-1u8qly9')
    try:
        number_of_pages = int(element.find_all('span')[-1].get_text())
    except AttributeError:
        number_of_pages = 1
    return number_of_pages


def parse_data():
    # Funkcja pobierająca dane o samochodach z linków i zapisująca je do pliku CSV
    print(datetime.datetime.now())
    links = get_links()
    cars = 0
    for link in links:
        time.sleep(random.randint(1, 5))
        number_of_pages = get_number_of_pages(link)
        if number_of_pages == 1:
            cars += scrap_page(link)
        else:
            scrap_page(link)
            link = link + '?page='
            for num in range(2, number_of_pages, 1):
                cars += scrap_page(link + str(num))
        print(datetime.datetime.now(), link, cars)


def scrap_page(url):
    # Funkcja pobierająca dane o samochodach z pojedynczej strony
    car = 0
    response = requests.get(url)
    html_code = response.text
    soup = BeautifulSoup(html_code, 'html.parser')

    elements = soup.find_all(class_='evg565y0')

    parsed_url = urlparse(url)
    path_segments = parsed_url.path.split("/")
    brand = path_segments[2]
    model = path_segments[3]

    for html in elements:
        price = int(html.find('span', class_='ooa-1bmnxg7 evg565y11').text.strip()[:-4].replace(" ", ""))
        information = html.find_all('li', class_='ooa-1k7nwcr e19ivbs0')
        information = [item.text.strip() for item in information]
        try:
            if information[2] in ('Elektryczny', 'Diesel', 'Benzyna', 'Benzyna+LPG', 'Hybryda'):
                information[0] = int(information[0].replace(" ", ""))
                information[2] = int(information[1][:-3].replace(" ", ""))
                information[1] = 0
            elif information[2] in ('Elektryczny', 'Hybryda'):
                information[0] = int(information[0].replace(" ", ""))
                information[1] = int(information[1][:-3].replace(" ", ""))
                information[2] = 0
            elif 'Niski przebieg' in information[0]:
                information[0] = int(information[1].replace(" ", ""))
                information[1] = int(information[2][:-3].replace(" ", ""))
                information[2] = int(information[3][:-3].replace(" ", ""))
            else:
                information[0] = int(information[0].replace(" ", ""))
                information[1] = int(information[1][:-3].replace(" ", ""))
                information[2] = int(information[2][:-3].replace(" ", ""))
            inf = [brand] + [model] + [price] + information[:3]
        except ValueError:
            print([brand] + [model] + [price] + information[:3])
            break

        with open('output.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(inf)
            car += 1
    return car


def del_dupl():
    # Funkcja usuwająca zduplikowane wiersze w pliku CSV
    df = pd.read_csv('output.csv')
    df.drop_duplicates(inplace=True)
    df.to_csv('new_data.csv', index=False)


def get_cars():
    # Funkcja tworząca słownik z markami i modelami samochodów
    filename = "new_data.csv"

    brands_models = {}

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            brand = row[0]
            model = row[1]

            if brand not in brands_models:
                brands_models[brand] = [(model, 1)]
            else:
                models = [m[0] for m in brands_models[brand]]
                if model not in models:
                    brands_models[brand].append((model, 1))
                else:
                    for i, (m, count) in enumerate(brands_models[brand]):
                        if m == model:
                            brands_models[brand][i] = (m, count + 1)
    return brands_models


def get_car():
    # Funkcja pobierająca markę i model samochodu od użytkownika
    filename = "new_data.csv"

    brands_models = get_cars()

    index = 1
    for brand in list(brands_models.keys()):
        print(f"{index}. {brand}")
        index += 1
    brand_i = input("Wpisz numer marki: ")
    brand = list(brands_models.keys())[int(brand_i) - 1]

    index = 1
    for model in brands_models[list(brands_models.keys())[int(brand_i) - 1]]:
        print(f"{index}. {model[0]}")
        index += 1
    model_i = input("Wpisz numer modelu: ")
    model = brands_models[list(brands_models.keys())[int(brand_i) - 1]][int(model_i) - 1][0]

    print(f"Wybrany samochód: {brand.upper()} {model.upper()}")
    return brand, model


def get_data_car(brand, model):
    # Funkcja pobierająca dane samochodów o określonej marce i modelu
    filename = "new_data.csv"
    matching_rows = []

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            if row[0] == brand and row[1] == model:
                matching_rows.append(row)

    return matching_rows


def get_price():
    # Funkcja prognozująca cenę samochodu na podstawie wprowadzonych danych
    brand, model = get_car()
    data = get_data_car(brand, model)
    year = input("Wpisz rok produkcji: ")
    mileage = input("Wpisz przebieg: ")
    volume = input("Wpisz pojemność skokowa silnika: ")

    x = np.array([[int(row[3]), int(row[4]), int(row[5])] for row in data])
    y = np.array([float(row[2]) for row in data])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    modelRFR = RandomForestRegressor()
    modelRFR.fit(x_train, y_train)

    new_data = np.array([[year, mileage, volume]])
    predicted_price = modelRFR.predict(new_data)

    print("-------------------------------------------------")
    print(f"Dla samochodu: {brand.upper()} {model.upper()}")
    print(f"Rok produkcji: {year}")
    print(f"Przebieg: {mileage} km")
    print(f"Pojemność skokową silnika: {volume} cm3")
    print(f"Oferujemy cenę: {predicted_price[0]} PLN")
    print("-------------------------------------------------")



def plot_brand_model_count(plot):
    if plot:
        brands_models = get_cars()
        brands = list(brands_models.keys())
        model_counts = [len(models) for models in brands_models.values()]

        plt.figure(figsize=(10, 6))
        plt.bar(brands, model_counts)
        plt.xlabel('Marka')
        plt.ylabel('Liczba modeli')
        plt.title('Zależność liczby modeli od marki')
        plt.xticks(rotation=45)
        plt.show()


if __name__ == "__main__":
    if not os.path.exists("output.csv"):
        parse_data()
        del_dupl()
    elif not os.path.exists("new_data.csv"):
        del_dupl()
    else:
        print("Witamy w oprogramowaniu do wyceny samochodów!")
        print("Dla jakiej marki i modelu pojazdu chciałbyś uzyskać cenę?")
        get_price()
    plot_brand_model_count(False)
