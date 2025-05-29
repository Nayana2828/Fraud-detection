{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "bc75ed93-7c48-4ca0-b199-bb5263511147",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "cc19f0be-a742-4d83-92b1-dddebc3c6a03",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "6511f444-f4b2-49d2-88b7-412c34ed6e21",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = joblib.load(\"fraud_detection_pipeline.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "db62db07-293a-4939-9e61-edeff3b7151e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:06:46.015 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:46.017 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title(\"Fraud Detection Prediction App\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "f8931c1b-1af2-4e2b-b8dd-d04de32e4a30",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:06:50.723 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:50.724 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.markdown(\"Please Enter the Transaction Details and Enter the Predict Button\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "c5bcf10b-a6cc-45d6-8b7a-c8bef8a3b5ad",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:06:52.137 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:52.139 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.divider()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "8fcb8bb2-54f5-460d-b1e5-6e016f6466a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:06:53.819 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.820 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.822 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.823 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.825 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.827 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.828 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.829 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.830 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.832 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.832 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:53.833 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "transaction_type = st.selectbox(\"Transaction Type\", [\"PAYMENT\",\"TRANSFER\",\"CASH_OUT\",\"DEPOSIT\"])\n",
    "amount = st.number_input(\"Amount\", min_value = 0.0, value = 1000.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "df9d152c-235d-4823-af03-e1f4a247e4fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:06:55.621 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:55.624 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:55.625 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:55.625 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:55.627 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:55.629 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "oldbalanceOrg = st.number_input(\"Old Balance (Sender)\", min_value = 0.0, value = 10000.0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "140c115d-eab5-488c-b0a1-2d1deb3e57a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:06:57.373 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:57.375 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:57.376 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:57.378 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:57.380 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:57.382 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "newbalanceOrig = st.number_input(\"New Balanace (Sender)\", min_value = 0.0, value = 9000.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "1678258e-9ee2-44d8-9fd4-fa1fd690ed29",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:06:58.721 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.722 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.723 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.724 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.726 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.728 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.729 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.731 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.732 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.734 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.735 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:06:58.736 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "oldbalanceDest = st.number_input(\"Old Balance (Receiver)\", min_value = 0.0, value = 0.0)\n",
    "newbalanceDest = st.number_input(\"New Balance (Receiver)\", min_value = 0.0, value = 0.0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "4a15cc4f-d35b-47e7-aaef-f81dabf2eec5",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_data = pd.DataFrame([{\n",
    "        \"type\":transaction_type,\n",
    "        \"amount\":amount,\n",
    "        \"oldbalanceOrg\":oldbalanceOrg,\n",
    "        \"newbalanceOrig\":newbalanceOrig,\n",
    "        \"oldbalanceDest\":oldbalanceDest,\n",
    "        \"newbalanceDest\":newbalanceDest\n",
    "    }])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "6e011bc3-51fa-44d4-af5e-9f6dc726b67d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:28:24.839 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:28:24.840 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:28:24.841 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:28:24.841 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:28:24.842 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if st.button(\"Predict\"):\n",
    "    input_data = pd.DataFrame([{\n",
    "        \"type\":transaction_type,\n",
    "        \"amount\":amount,\n",
    "        \"oldbalanceOrg\":oldbalanceOrg,\n",
    "        \"newbalanceOrig\":newbalanceOrig,\n",
    "        \"oldbalanceDest\":oldbalanceDest,\n",
    "        \"newbalanceDest\":newbalanceDest\n",
    "    }])\n",
    "        \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "25ef9abc-f157-40d4-a34d-9d8267481ed0",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = model.predict(input_data)[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "43181d48-6efd-429b-a30a-85dbb1fc0204",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:28:38.256 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:28:38.257 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.subheader(f\"Prediction:'{int(prediction)}'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "0b3020f5-bbc0-4ba4-bf23-110a336465a9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-29 10:28:47.157 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-29 10:28:47.158 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if prediction == 1:\n",
    "    st.error(\"This transaction can be fraud\")\n",
    "else:\n",
    "    st.success(\"This transaction looks like it is not a fraud\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d97404b1-5822-4144-976e-8ecca09c70e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "!streamlit run fraud_detection.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43e52977-20c0-476e-ad5f-8f36038dfffe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
