import numpy as np
import argparse


mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949


vocab_reverse = [
           'A', # 0
           'R', # 1
           'N', # 2
           'D', # 3
           'E', # 5
           'Q', # 6
           'G', # 7
           'H', # 8
           'I', # 9
           'L', # 10
           'K', # 11
           'M', # 12
           'F', # 13
           'P', # 14
           'S', # 15
           'T', # 16
           'W', # 17
           'Y', # 18
           'V', # 19
           'C',
           '!',# N(Deamidation)
           '@', # C(+57.02) C(Carbamidomethylation)
           '#',# Q(Deamidation)
           '~',# M(Oxidation)
           '?',
           '^',#[Acetyl]-
           '&',#[Carbamyl]-
           '$',#[Ammonia-loss]-
                ]

# vocab_reverse = _START_VOCAB + vocab_reverse
vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
vocab_size = len(vocab_reverse)

mass_AA = {
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           'D': 115.02694, # 3
           #~ 'C(Carbamidomethylation)': 103.00919, # 4
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
           'C': 103.01019, #C
           '!': 115.02695,# N(Deamidation)
           '@': 160.03065, # C(+57.02) C(Carbamidomethylation)
           '#': 129.0426,# Q(Deamidation)
           '~': 147.0354,# M(Oxidation)
           "?": 42.011+114.04293,
           "^": 42.011,   #[Acetyl]-
           "&": 43.006,   #[Carbamyl]-
           "$": -17.03, #[Ammonia-loss]-
          }
mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]


class WorkerTest(object):
  def __init__(self,target_file,pep_len):
    self.MZ_MAX = 40000
    self.target_file = target_file
    self.predicted_file = target_file
    print("target_file = {0:s}".format(self.target_file))
    print("predicted_file = {0:s}".format(self.predicted_file))

    self.target_dict = {}
    self.predicted_list = []
    self.num_wrong=[]
    self.pep_len=pep_len
  
  def test_accuracy(self):
    self._get_target()
    target_dict_db = self.target_dict
    target_dict_db_mass = {}
    for feature_id, peptide in target_dict_db.items():
      if self._compute_peptide_mass(peptide) <= self.MZ_MAX:
        target_dict_db_mass[feature_id] = peptide
    target_count_db_mass = len(target_dict_db_mass)
    target_len_db_mass = sum([len(x) for x in target_dict_db_mass.values()])

    self._get_predicted()

    predicted_count_mass_db = 0
    predicted_len_mass_db = 0
    recall_AA_total = 0.0
    recall_peptide_total = 0.0
    pep_aa_acc=[]

    for index, predicted in enumerate(self.predicted_list):

      feature_id = predicted["feature_id"]


      if feature_id in target_dict_db_mass:
        target = target_dict_db_mass[feature_id]
        target_len= len(target)
        predicted_sequence=predicted["sequence"]
        if predicted_sequence==[]:
          continue

        try:
          predicted_AA_id = [vocab[x] for x in predicted_sequence]
          target_AA_id = [vocab[x] for x in target]
        except Exception as e:
          print(predicted_sequence)
          print(target)
          exit(1)
        predicted_count_mass_db += 1
        recall_AA = self._match_AA_novor(target_AA_id, predicted_AA_id)

        recall_AA_total += recall_AA
        self.num_wrong.append(target_len-recall_AA)
        pep_aa_acc.append(recall_AA/target_len)
        if recall_AA == target_len:
          recall_peptide_total += 1
        predicted_len_mass_db += len(predicted_sequence)

        recall_AA = "{0:d}".format(recall_AA)
        target_len = "{0:d}".format(target_len)

      else:
        print("no predicted peptide id in target!")
        exit(1)
    print(f"total peptides before filter: {len(self.target_dict)}")
    print("predicted_peptides: {0:d}".format(predicted_count_mass_db))
    print("predicted_AAs: {0:d}".format(predicted_len_mass_db))
    print("target_peptides: {0:d}".format(target_count_db_mass))
    print("target_AAs: {0:d}".format(target_len_db_mass))
    print(f"correct_AAs: {recall_AA_total}")
    print(f"correct_pep: {recall_peptide_total}")

    print("recall_AA = {0:.3f}".format(recall_AA_total / target_len_db_mass))
    print("recall_peptide = {0:.3f}".format(recall_peptide_total / target_count_db_mass))

    print("precision_AA  = {0:.3f}".format(recall_AA_total / predicted_len_mass_db))
    print("precision_peptide  = {0:.3f}".format(recall_peptide_total / predicted_count_mass_db))
  
    final_acc = sum(pep_aa_acc)/len(pep_aa_acc)
    print("AA accuracy at peptide level:{0:.3f}".format(final_acc))

  def _compute_peptide_mass(self, peptide):
    peptide_mass = (mass_N_terminus
                    + sum(mass_AA[aa] for aa in peptide)
                    + mass_C_terminus)

    return peptide_mass


  def _get_predicted(self):
    print("WorkerTest._get_predicted()")

    predicted_list = []
    i=0
    with open(self.predicted_file, 'r') as handle:
      handle.readline()
      for line in handle:
        line_split = line.strip().split("\t")
        predicted = {}
        predicted["feature_id"] = str(i)
        if line_split[1]: # not empty sequence
          predicted["sequence"] = [x for x in line_split[1]]
        else: 
          predicted["sequence"] = []
        predicted_list.append(predicted)
        i+=1
    self.predicted_list = predicted_list



  def _get_target(self):
    print("WorkerTest._get_target()")

    target_dict = {}
    i=0
    with open(self.target_file, 'r') as handle:
      # header_line = handle.readline()
      for line in handle:
        line = line.strip().split("\t")
        feature_id = str(i)
        raw_sequence = line[0] #feature file seq columne
        peptide = raw_sequence
        target_dict[feature_id] = peptide
        i+=1
    self.target_dict = target_dict

  def _match_AA_novor(self, target, predicted):
    num_match = 0
    target_len = len(target)
    predicted_len = len(predicted)
    target_mass = [mass_ID[x] for x in target]
    target_mass_cum = np.cumsum(target_mass)
    predicted_mass = [mass_ID[x] for x in predicted]
    predicted_mass_cum = np.cumsum(predicted_mass)
  
    i = 0
    j = 0
    while i < target_len and j < predicted_len:
      if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
        if abs(target_mass[i] - predicted_mass[j]) < 0.1:
          num_match += 1
        i += 1
        j += 1
      elif target_mass_cum[i] < predicted_mass_cum[j]:
        i += 1
      else:
        j += 1

    return num_match


def calppm(mz1,mz2, ppm):
  if (abs(mz1-mz2)/mz1)*1e6 <= ppm:
    return True
  else:
    return False

def readresultppm(inputfile, outputfile, ppm):
  data=[]
  inputf=open(inputfile,"r")
  headers=inputf.readline().strip().split("\t")
  header={v:k for k,v in enumerate(headers)}

  while True:
    line=inputf.readline()
    if not line:
      break
    tmp=line.rstrip().split("\t") 
    aa_s=float(tmp[header["pep_score"]])
    rev_aa_s=float(tmp[header["rev_pep_score"]])
    true_mz=float(tmp[header["true_mz"]])
    if aa_s >= rev_aa_s:
      pred_mz=float(tmp[header["forward_mz"]])
      if calppm(true_mz, pred_mz, ppm):
        data.append(tmp[header["True_seq"]].replace("I","L")+"\t"+tmp[header["seq"]].replace("I","L")+"\t"+tmp[header["aa_score"]]+"\t"+str(aa_s)+"\tF\t"+tmp[header["true_mz"]]+"\t"+tmp[header["charge"]]+"\t"+tmp[header["forward_mz"]]+"\n")
      else:
        data.append(tmp[header["True_seq"]].replace("I","L")+"\t\t-1,\t-1\tF\t"+tmp[header["true_mz"]]+"\t"+tmp[header["charge"]]+"\t0.0\n")
    else:
      pred_mz=float(tmp[header["rev_mz"]])
      if calppm(true_mz, pred_mz, ppm):
        data.append(tmp[header["True_seq"]].replace("I","L")+"\t"+tmp[header["rev_seq"]].replace("I","L")+"\t"+tmp[header["rev_aa_score"]]+"\t"+str(rev_aa_s)+"\tR\t"+tmp[header["true_mz"]]+"\t"+tmp[header["charge"]]+"\t"+tmp[header["rev_mz"]]+"\n")
      else:
        data.append(tmp[header["True_seq"]].replace("I","L")+"\t\t-1,\t-1\tR\t"+tmp[header["true_mz"]]+"\t"+tmp[header["charge"]]+"\t0.0\n")
  inputf.close()
  outf=open(outputfile,"w")
  outf.write("true_seq\tpred_seq\taa_scores\tpep_score\tlabel\ttrue_mz\tcharge\tpred_mz\n")
  for tmp in data:
    outf.write(tmp)

  outf.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--in_file",type=str,default="",help='the input result file path')
  parser.add_argument("--ppm",type=int,default=30,help='the ppm for filtering between feature mz and predicted peptide mz')
  args = parser.parse_args()

  readresultppm(args.in_file, "after_"+args.in_file, args.ppm)
  testing = WorkerTest("after_"+args.in_file, pep_len=30)
  testing.test_accuracy()