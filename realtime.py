import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import joblib
# import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from pyparsing import empty


from keras.models import load_model


test =px.data.tips()


st.set_page_config(
    page_title="Dissolution Tank Condition",
    layout="wide",
)


# Data Prepare
FilePath = "D:/02_Works/18_KAMP/melting_tank_stream.csv"
scaler_call = joblib.load("mx_rscaler.pkl") 
model_call = load_model('melting_tank_pretrained_model.h5')

@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(FilePath)

df = get_data()
df_Length = df.shape[0]




# dashboard title

st.title("Dissolution Tank Condition")



placeholder_1 = st.empty()
placeholder_2 = st.empty()
# near real-time simulation
Pie_Value = [5, 1]
for seconds in range(df_Length):
    
    ndf = df.iloc[seconds:seconds+30]
    result_mapping = {
        "OK":1,
        "NG":0
    }
    ndf.loc[:,"TAG"]=ndf.TAG.map(result_mapping)
    
    
    
    with placeholder_1.container():
        Condition_Record, Condition_Est, Dummy_Area = st.columns((1,1,1))
        with Condition_Record:
            st.markdown("## Recorded TAG")
            Tag_Value=ndf["TAG"][seconds+29]
            if Tag_Value==1:
                st.success("OK")
                Pie_Value[0] = Pie_Value[0] + 1
            else:
                st.error("NG")
                Pie_Value[1] = Pie_Value[1] + 1 
        with Condition_Est:
            
            st.markdown("## Estimated Condition")
            ndf.drop(['MELT_WEIGHT', 'TAG'], axis=1)
            
            new_x_df = pd.concat([ndf["MELT_TEMP"][20:],ndf["MOTORSPEED"][20:]] ,axis=1)
            new_x_df_scale = scaler_call.transform(new_x_df)  
            new_x_df_scale = new_x_df_scale.reshape(1,10,2)
            Tag_Est = model_call.predict(new_x_df_scale) 
            
            
            if Tag_Est > 0.5:
                st.success("OK")
                # Pie_Value[0] = Pie_Value[0] + 1
            else:
                st.error("NG")
                # Pie_Value[1] = Pie_Value[1] + 1 
            
        
    with placeholder_2.container():
        
        Tank_Graph, Fail_Ratio, Tank_Data = st.columns((2,1,1))
        with Tank_Graph:
            st.markdown("## Tank Condition")
            fig = make_subplots(rows=4, cols=1,
                                subplot_titles=("Melting Temperature", "Motor Speed", "Melt Weight", "INSP"))
            # fig.update_layout(height = 1400)
            fig.update_layout(margin=dict(l = 20,
                                         r=20,
                                         b=50,
                                         t=20,
                                         pad = 4))
            fig.add_trace(go.Scatter(x=ndf.NUM, y=ndf.MELT_TEMP,
                                        mode = "lines",
                                        name = 'Temp.'),
                                        row=1, col=1)
            fig.add_trace(go.Scatter(x=ndf.NUM, y=ndf.MOTORSPEED,
                                        mode = "lines",
                                        name = 'Motor Speed'),
                                        row=2, col=1)
            fig.add_trace(go.Scatter(x=ndf.NUM, y=ndf.MELT_WEIGHT,
                                        mode = "lines",
                                        name = 'WEIGHT'),
                                        row=3, col=1)
            fig.add_trace(go.Scatter(x=ndf.NUM, y=ndf.INSP,
                                        mode = "lines",
                                        name = 'INSP'),
                                        row=4, col=1)
            
            
            fig.add_annotation(x=ndf.NUM[seconds+29], y=ndf.MELT_TEMP[seconds+29],
                                text = 'CV',
                                showarrow = True,
                                arrowhead= 1,
                                row=1, col=1)
            fig.add_annotation(x=ndf.NUM[seconds+29], y=ndf.MOTORSPEED[seconds+29],
                                text = 'CV',
                                showarrow = True,
                                arrowhead= 1,
                                row=2, col=1)
            fig.add_annotation(x=ndf.NUM[seconds+29], y=ndf.MELT_WEIGHT[seconds+29],
                                text = 'CV',
                                showarrow = True,
                                arrowhead= 1,
                                row=3, col=1)
            fig.add_annotation(x=ndf.NUM[seconds+29], y=ndf.INSP[seconds+29],
                                text = 'CV',
                                showarrow = True,
                                arrowhead= 1,
                                row=4, col=1)
            st.plotly_chart(fig, use_container_width= True)
            
        with Fail_Ratio:
            st.markdown("## Estimated NG Ratio")
            labels = ['OK', 'NG']
            
            fig = go.Figure(data =[go.Pie(labels = labels, values= Pie_Value, hole=0.3)])
            st.plotly_chart(fig, use_container_width= True)
            
                
        with Tank_Data:
            st.markdown("## Recorded Data View")
            view_NDF = ndf.drop(['STD_DT'], axis=1)
            st.dataframe(view_NDF[20:])

        

        
        time.sleep(1)