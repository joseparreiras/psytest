import pandas as pd

url: str = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/02d69a38-97f2-45f8-941d-4e4c5b50dea7/ie_data.xls?ver=1743773003799"
)

data: pd.DataFrame = (
    pd.read_excel(
        url,
        sheet_name="Data",
        skiprows=7,
        usecols=["Date", "P", "D", "E", "CAPE"],
        skipfooter=1,
        dtype={"Date": str, "P": float},
    )
    .rename(
        {
            "P": "sp500",
            "CAPE": "cape",
            "Date": "date",
            "D": "dividends",
            "E": "earnings",
        },
        axis=1,
    )
    .assign(
        date=lambda x: pd.to_datetime(x["date"].str.ljust(7, "0"), format="%Y.%m"),
    )
    .set_index("date", drop=True)
)

data.to_csv("tests/data/shiller_data.csv", index=True)
