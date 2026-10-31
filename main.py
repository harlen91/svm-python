import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#membaca model pkl
model = joblib.load("model.pkl")

#format data masukkan sesuai fitur data csv
class DataMahasiswa(BaseModel):
    jam_belajar_per_minggu : float
    kehadiran_persen : float
    rata_rata_tugas : float
    nilai_uts : float

#membuat end poin/alamat akses api
@app.get('/')
def read_root():
    return {"message" : "Prediksi Kelulusan Mahasiswa"}

@app.post("/predict")
def predict(data : DataMahasiswa):
    fitur = [[data.jam_belajar_per_minggu, data.kehadiran_persen, data.rata_rata_tugas, data.nilai_uts]]
    hasil = model.predict(fitur)

    return {"Hasil":int(hasil[0])}