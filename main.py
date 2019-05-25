from triathon_ml.prediction import *

pred = prediction()
pred.generate_a_dataset()
pred.train_model()
pred.load_model_and_pred()