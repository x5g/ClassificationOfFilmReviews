acc = [
0.5203020572662354,
0.6764176487922668,
0.7850785255432129,
0.8268826603889465,
0.8517351746559143,
0.8702370524406433,
0.8858385682106018,
0.9015901684761047,
0.914791464805603,
0.9266926646232605,
0.9362436532974243,
0.9459946155548096]
val_acc = [
0.5158997774124146,
0.8187955021858215,
0.8331853151321411,
0.8385148048400879,
0.8445549607276917,
0.8433114290237427,
0.8429561257362366,
0.8427784442901611,
0.8433114290237427,
0.84562087059021,
0.8399360179901123,
0.8406466245651245]
loss = [
1.5432657950615236,
0.7010270053356311,
0.5163566627220125,
0.45166202675033684,
0.4169329752122322,
0.3864818722060328,
0.358524215198753,
0.33330841018821206,
0.3086123798045889,
0.28480633526089694,
0.26327509169328667,
0.240185011309771]
val_loss = [
0.9288353713076754,
0.5369245947137948,
0.439178810288627,
0.41637633871938984,
0.4052381435451213,
0.39904334815574766,
0.39425602215372835,
0.3942812527924495,
0.39734860925378795,
0.4040254990919845,
0.40756674749132454,
0.42959259286573165]

epochs = len(acc)
# modelResultsFig = 'textCNN_result_8.png'


from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('textCNN_result_8.pdf')

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(acc, color='blue', linestyle='-', label='Train accuracy')
lns2 = ax.plot(val_acc, color='orange', linestyle='-', label='Validation accuracy')
ax2 = ax.twinx()
lns3 = ax2.plot(loss, color='red', linestyle='-', label='Train loss')
lns4 = ax2.plot(val_loss, color='green', linestyle='-', label='Validation loss')
lns = lns1 + lns2 + lns3 + lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper right')
# ax.legend(loc=0)
ax.grid()
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.10)
ax.xaxis.set_major_locator(x_major_locator)
ax.set_xlim(0, epochs - 1)
ax.set_ylim(0.5, 1.1)
ax.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_major_locator(MultipleLocator(0.15))
ax2.set_ylabel("Loss")
ax2.set_ylim(0.15, 1.65)
# ax2.legend(loc=0)
plt.title('Training and validation accuracy and loss')
# plt.show()
# plt.savefig(modelResultsFig)

plt.tight_layout()
pdf.savefig()
plt.close()
pdf.close()
