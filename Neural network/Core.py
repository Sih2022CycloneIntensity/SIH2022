from Model import RPN, Base, Classifier, RoiPooling
from keras.models import Model
from keras.layers import Input

AnchorCount = 15
PoolHeightClassifier = 7
PoolWidthClassifier = 7
PoolHeightMask = 7
PoolWidthMask = 7
NumRoI = 20

I0 = Input(shape=(224, 224, 3))
I1 = Input(shape=(None, 4))

L1 = Base(I0)

R1 = RPN(L1, AnchorCount)
print(R1[0].shape, R1[1].shape)
ROI = RoiPooling(R1[0], I1, PoolHeightClassifier, PoolWidthClassifier)

C1 = Classifier(ROI, 15)

M1 = Model(inputs=[I0, I1], outputs=C1)

M1.summary(line_length=200)


def ModelRPN():
    model_rpn = Model(inputs=[I0, I1], outputs=R1)
    return model_rpn


ModelRPN().summary()
