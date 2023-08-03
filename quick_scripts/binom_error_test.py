import ROOT
import numpy as np

ROOT.TH1.SetDefaultSumw2()

h1 = ROOT.TH1F("test1", "test1", 1, 0, 1)
h2 = ROOT.TH1F("test2", "test2", 1, 0, 1)
h3 = ROOT.TH1F("test3", "test3", 1, 0, 1)
h4 = ROOT.TH1F("test4", "test4", 1, 0, 1)
# h5 = ROOT.TH1F("test5", "test5", 1, 0, 1)

n1 = 10
n2 = 20

for i in range(n1):
    h1.Fill(0.5)

for i in range(n2):
    h2.Fill(0.5)

h3.Divide(h1, h2, 1, 1, "b")
h4.Divide(h1, h2)

h = h1.Clone()
tgraph = ROOT.TGraphAsymmErrors(h1, h2, "cl=0.683 b(1,1) mode")

print(f"h1 content: {h1.GetBinContent(1)}")
print(f"h1 error: {h1.GetBinError(1)}")
print(f"h2 content: {h2.GetBinContent(1)}")
print(f"h2 error: {h2.GetBinError(1)}")
print(f"binom content: {h3.GetBinContent(1)}")
print(f"binom error: {h3.GetBinError(1)}")
print(f"h4 content: {h4.GetBinContent(1)}")
print(f"h4 error: {h4.GetBinError(1)}")
print(f"bayes error: {tgraph.GetErrorY(0)}")

# manual error
b1 = h1.GetBinContent(1)
b2 = h2.GetBinContent(1)
eff = b1 / b2
b1sq = b1**2
b2sq = b2**2
e1sq = h1.GetBinError(1) ** 2
e2sq = h2.GetBinError(1) ** 2

err = np.sqrt(eff * (1 - eff) * b2) / b2
# err = abs(((1.0 - 2.0 * b1 / b2) * e1sq + b1sq * e2sq / b2sq) / b2sq)
print(f"manual binom. error : {err}")

err = (e1sq * b2sq + e2sq * b1sq) / (b2sq * b2sq)
print(f"manual error : {err}")
