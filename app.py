import streamlit as st
import pickle
import pandas as pd

model_bundle = pickle.load(open("rf_fraud_model.pkl", "rb"))

model = model_bundle["model"]
encoder_method = model_bundle["encoder_method"]
encoder_provider = model_bundle["encoder_provider"]
scaler = model_bundle["scaler"]
feature_columns = model_bundle["feature_columns"]

st.title("🔍 Fraud Detection Demo App")
st.write("Masukkan data transaksi untuk melihat prediksi fraud.")

transaction_amount = st.number_input("Transaction Amount", min_value=1, value=120000)
payment_method_name = st.selectbox("Payment Method", encoder_method.classes_)
payment_provider_name = st.selectbox("Payment Provider", encoder_provider.classes_)
processing_seconds = st.number_input("Processing Seconds", min_value=0.0, value=5.0)
buyer_seller_tx_count = st.number_input("Buyer–Seller Tx Count", min_value=0, value=5)
buyer_total_tx = st.number_input("Buyer Total Tx", min_value=0, value=10)
buyer_avg_amount = st.number_input("Buyer Average Amount", min_value=0, value=50000)
buyer_promo_count = st.number_input("Buyer Promo Count", min_value=0, value=2)
seller_total_tx = st.number_input("Seller Total Tx", min_value=0, value=120)
seller_repeat_buyer = st.number_input("Repeat Buyer Count", min_value=0, value=3)

if st.button("Predict Fraud"):

    df_input = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "payment_method_name": payment_method_name,
        "payment_provider_name": payment_provider_name,
        "processing_seconds": processing_seconds,
        "buyer_seller_tx_count": buyer_seller_tx_count,
        "buyer_total_tx": buyer_total_tx,
        "buyer_avg_amount": buyer_avg_amount,
        "buyer_promo_count": buyer_promo_count,
        "seller_total_tx": seller_total_tx,
        "seller_repeat_buyer": seller_repeat_buyer
    }])

    df_input["payment_method_name"] = encoder_method.transform(
        df_input["payment_method_name"]
    )
    df_input["payment_provider_name"] = encoder_provider.transform(
        df_input["payment_provider_name"]
    )

    df_input["rule_big_amount"] = (
        df_input["transaction_amount"] > (df_input["buyer_avg_amount"] * 3)
    )

    df_input["rule_fast"] = (df_input["processing_seconds"] < 1)

    df_input["rule_new_buyer_big_tx"] = (
        (df_input["buyer_total_tx"] == 1) &
        (df_input["transaction_amount"] > 10000000)
    )

    df_input["rule_abnormal_ratio"] = (
        (df_input["buyer_total_tx"] + 1) /
        (df_input["seller_total_tx"] + 1)
    ) > 50

    df_input["rule_promo_abuse"] = (
        df_input["buyer_promo_count"] > 10
    )

    scale_cols = [
        "transaction_amount",
        "processing_seconds",
        "buyer_avg_amount"
    ]
    df_input[scale_cols] = scaler.transform(df_input[scale_cols])

    data_input = df_input[feature_columns]

    prob = model.predict_proba(data_input)[0][1]
    pred = 1 if prob >= 0.30 else 0

    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Probabilitas Fraud:** `{prob:.3f}`")
    st.write(f"**Threshold:** `0.30`")

    if pred == 1:
        st.error("⚠️ Transaksi berpotensi FRAUD!")
    else:
        st.success("✅ Transaksi aman / normal.")

    st.json(data_input.to_dict(orient="records")[0])
