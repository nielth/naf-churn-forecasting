import requests
import logging
import json
import os

from tqdm import tqdm
import pandas as pd


def save_mapping(mapping, filename="car_mapping_checkpoint.json"):
    with open(filename, "w") as f:
        json.dump(mapping, f, indent=2)


def load_mapping(filename="car_mapping_checkpoint.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    abonnementsprodukter__historikk_df = pd.read_csv(
        "received_csvs/abonnementsprodukter__historikk.csv"
    )

    df_parts = [
        pd.read_csv(f"received_csvs/eierverifiseringer__historikk_part{i}.csv")
        for i in range(1, 7)
    ]
    eierverifiseringer__historikk_df = pd.concat(df_parts, ignore_index=True)

    stg_dynamics__contacts_df = pd.read_csv("received_csvs/stg_dynamics__contacts.csv")

    filtered_df = stg_dynamics__contacts_df[
        stg_dynamics__contacts_df["contact_rk"].isin(
            abonnementsprodukter__historikk_df["subscriptionitem__contact_rk"]
        )
    ]

    temp = eierverifiseringer__historikk_df[
        eierverifiseringer__historikk_df["contact_rk"].isin(filtered_df["contact_rk"])
    ]

    temp["modell"] = temp["modell"].astype(str).str.split(r"[ ,]").str[0]

    cleaned = (
        temp[["merke", "modell", "årsmodell", "drivstoff"]]
        .dropna()
        .drop_duplicates()
        .astype({"merke": str, "modell": str, "årsmodell": int, "drivstoff": str})
    )

    cars = cleaned.apply(
        lambda row: [
            f"{row['merke']} {row['modell']} {row['drivstoff']}",
            row["årsmodell"],
        ],
        axis=1,
    ).tolist()

    logging.basicConfig(
        filename="car_scraper.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    url = "https://www.finn.no/mobility/search/api/search/SEARCH_ID_CAR_USED?q={}&sales_form=1&page={}&year_to={}&year_from={}"
    car_mapping = load_mapping()

    try:
        for i, car in enumerate(tqdm(cars, desc="Processing cars")):
            car_key = f"{car[0]} {car[1]}"
            if car_key in car_mapping:
                continue

            prices = []
            to = str(car[1])
            from_ = str(car[1])

            try:
                resp = requests.get(url.format(car[0], "1", to, from_))
                resp.raise_for_status()
                rjson = resp.json()
            except Exception as e:
                logging.error(f"Request/JSON error for {car}: {e}")
                continue

            docs = rjson.get("docs")
            if not docs:
                logging.info(f"NoneType: {car}")
                continue

            if len(docs) < 1:
                try:
                    resp = requests.get(
                        url.format(car[0], "1", str(car[1] + 1), str(car[1] - 1))
                    )
                    resp.raise_for_status()
                    rjson = resp.json()
                    docs = rjson.get("docs", [])
                except Exception as e:
                    logging.error(f"Fallback fetch failed for {car}: {e}")
                    continue

                if len(docs) < 1:
                    logging.info(f"EMPTY: {car}")
                    continue

            for item in docs:
                price_dict = item.get("price")
                if not price_dict:
                    continue
                price_int = price_dict.get("amount", 0)
                if price_int < 1:
                    continue
                prices.append(int(price_int))

            try:
                pages = int(rjson["metadata"]["paging"]["last"])
            except (KeyError, TypeError, ValueError) as e:
                logging.warning(f"Pagination info missing for {car}: {e}")
                pages = 1

            if pages > 1:
                for page in range(2, pages + 1):
                    try:
                        resp = requests.get(
                            url.format(
                                car[0], str(page), str(car[1] + 1), str(car[1] - 1)
                            )
                        )
                        resp.raise_for_status()
                        page_json = resp.json()
                        docs = page_json.get("docs", [])
                    except Exception as e:
                        logging.error(f"Error on page {page} for {car}: {e}")
                        continue

                    for item in docs:
                        price_dict = item.get("price")
                        if not price_dict:
                            continue
                        price_int = price_dict.get("amount", 0)
                        if price_int < 1:
                            continue
                        prices.append(int(price_int))

            car_mapping[car_key] = prices

            if i % 10 == 0:
                save_mapping(car_mapping)

    finally:
        save_mapping(car_mapping, "car_mapping_final.json")
