import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('AutoLR_model.pkl', 'rb'))

def predict_price(horse_power, curb_weight, engine_size, highway_mpg):
    input = np.array([[horse_power, curb_weight, engine_size, highway_mpg]]).astype(np.float)
    prediction = model.predict(input)
    return float(prediction)

def main():
    st.title('Car price prediction')

    horse_power = st.text_input('What is your horse power?')
    curb_weight = st.text_input('What is your curb weight?')
    engine_size = st.text_input('what is your engine size?')
    highway_mpg = st.text_input('what is your highway mpg?')

    if st.button('Predict'):
        output = predict_price(horse_power, curb_weight, engine_size, highway_mpg)
        st.success('Selling price is approxmately', format(round(output,2)))

if __name__ =='__main__':
    main()
