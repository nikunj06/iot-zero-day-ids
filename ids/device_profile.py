
import pandas as pd


def build_device_profiles(data):
    required = {"src", "flow_id", "predicted", "dst"}

    if not required.issubset(data.columns):
        raise ValueError("Missing required device profiling columns")

    df = data.copy()
    df = df.dropna(subset=["src"])
    df["src"] = df["src"].astype(str)

    df["is_alert"] = df["predicted"].isin(
        ["unknown", "lab_alert"]
    )

    profiles = df.groupby("src").agg(
        total_flows=("flow_id", "count"),
        alert_count=("is_alert", "sum"),
        unique_destinations=("dst", "nunique")
    ).reset_index()

    profiles["alert_rate"] = (
        profiles["alert_count"] / profiles["total_flows"]
    )

    rate_score = (
        profiles["alert_rate"].clip(upper=0.10) / 0.10
    ) * 60

    count_score = (
        profiles["alert_count"].clip(upper=20) / 20
    ) * 40

    profiles["risk_score"] = (
        rate_score + count_score
    ).round(1)

    def risk_level(row):
        if row["total_flows"] < 10:
            return "Insufficient data"

        if row["risk_score"] >= 60:
            return "High"

        if row["risk_score"] >= 30:
            return "Medium"

        return "Low"

    profiles["risk_level"] = profiles.apply(
        risk_level,
        axis=1
    )

    profiles = profiles.sort_values(
        "risk_score",
        ascending=False
    )

    return profiles
