import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from numpy.typing import ArrayLike
from pyteomics import proforma, mass
import sys


model_type = torch.float32

HYDROGEN = 1.007825035
OXYGEN = 15.99491463
H2O = 2 * HYDROGEN + OXYGEN
PROTON = 1.00727646688
C13 = 1.003355


mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

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
           '!': 115.02695,# N(Deamidation)
           '@': 160.03065, # C(+57.02) C(Carbamidomethylation)
           '#': 129.0426,# Q(Deamidation)
           '~': 147.0354,# M(Oxidation)
           '$':0.0
          }#num:26

seq_size=len(mass_AA)
print(seq_size)
conv_mod={
    "C(+57.02)":"@",
    "M(+15.99)":"~",
}
id_to_seq={i+1:k for i, k in enumerate(sorted(list(mass_AA.keys())))}
seq_to_id={k:i+1 for i, k in enumerate(sorted(list(mass_AA.keys())))}
print(id_to_seq)

MSKB_TO_UNIMOD = {
    "+42.011": "[Acetyl]-",
    "+43.006": "[Carbamyl]-",
    "-17.027": "[Ammonia-loss]-",
    "+43.006-17.027": "[+25.980265]-",  # Not in Unimod
    "M+15.995": "M[Oxidation]",
    "N+0.984": "N[Deamidated]",
    "Q+0.984": "Q[Deamidated]",
    "C+57.021": "C[Carbamidomethyl]",
}



def decoderSeq(token,aa_score):
    seq=""
    score=[]
    if aa_score:
        for tt, ss in zip(token,aa_score):
            if tt==0 or tt==seq_to_id["$"]:
                break
            else:
                seq+=id_to_seq[tt]
                score.append(ss)
    else:
        for tt in token:
            if tt==0 or tt==seq_to_id["$"]:
                break
            else:
                seq+=id_to_seq[tt]

    return seq, score


def readmgfOffset(file:str)->dict:
    result={}
    start=0
    end=0
    scan=""
    with open(file,"r") as f:
        line=f.readline()
        while True:
            line=f.readline()
            if not line:
                break
            if line.find("SCANS=")!=-1:
                scan=line[len("SCANS="):-1]
            elif line.find("END ION")!=-1:
                end=f.tell()
                result[scan]=[start,end]
                start=end

    f.close()
    return result

def readSeq(file:str)->list[list]:
    result=[]
    ff=open("file","r")
    line=ff.readline()
    while True:
        line=ff.readline()
        if not line:
            break
        seq=line.split(",")[4]
        seq=seq.replace("C(+57.02)",")20.75+(C")
        seq=seq.replace("M(+15.99)",")99.51+(M")
        rev_seq=seq[::-1]
        result.append([seq,rev_seq])
    ff.close()
    return result


class DIADataset(Dataset):
    def __init__(self, mgf_file:str, feat_file:str, ms_file_offset:dict, opt:object, mz_range:int=-1, rt_half_range:int=11):
        self.opt=opt
        self.ms_file_offset=ms_file_offset
        self.mgf_file=mgf_file
        self.mz_range=mz_range
        self.rt_half_range=rt_half_range
        self.feats=[]
        ff=open(feat_file,"r")
        line=ff.readline()
        while True:
            line=ff.readline()
            if not line:
                break
            tmp=line.strip().split(",")
            try:
                self.feats.append([tmp[0],float(tmp[1]), int(float(tmp[2])), float(tmp[3]), tmp[4], tmp[5], tmp[6], float(tmp[7])])
            except Exception as e:
                print(e)
                print(tmp)
                ff.close()
                sys.exit(1)
        ff.close()
        self.index={k:i+1 for i, k in enumerate(sorted(list(mass_AA.keys())))}
        
    def __len__(self):
        return len(self.feats)
    def __getitem__(self, idx):
        ms1_spect=[]
        ms2_spect=[]
        peak_label=[]
        fragment_label=[]
        ms1_norm=[]
        ff=open(self.mgf_file,"r")
        _, mz, z, rt, seq, ms2, ms1, area=self.feats[idx]

        if seq != "":
            seq=seq.replace("C(+57.02)", conv_mod["C(+57.02)"])
            seq=seq.replace("M(+15.99)", conv_mod["M(+15.99)"])
            try:
                token_seq=[self.index[aa] for aa in seq]
            except Exception as e:
                print(e)
                print(seq)
                sys.exit(1)
            rev_token_seq=token_seq[::-1]
        else:
            token_seq = []
            rev_token_seq = []

        tmp=ms1.split(";")
        ms1_elu_rt=[]
        ms1_elu_int=[]
        for aa in tmp:
            tmprt, tmpint=aa.split(":")
            ms1_elu_rt.append(float(tmprt))
            ms1_elu_int.append(float(tmpint))
        n=len(ms1_elu_rt)
        ms1_elu_rt=np.array(ms1_elu_rt)
        ms1_elu_int=np.sqrt(np.array(ms1_elu_int))
        rt_diff=np.abs(ms1_elu_rt-rt)
        location=np.argmin(rt_diff)
        start=0
        end=n
        if 2*self.rt_half_range+1<n:
            if location-self.rt_half_range<0:
                start=0 
                end=start+2*self.rt_half_range+1
            elif location+self.rt_half_range>=n:
                end=n
                start= n-2*self.rt_half_range-1
            else:
                start=location-self.rt_half_range
                end=location+self.rt_half_range+1

        ms2_scan=ms2.split(";")

        for tmp_idx in range(start,end):
            ms1_spect.append([mz, ms1_elu_int[tmp_idx], ms1_elu_rt[tmp_idx]])
            ms1_norm.append(ms1_elu_int[tmp_idx])
            ff.seek(self.ms_file_offset[ms2_scan[tmp_idx]][0])
            ms2_rt=0.0
            while True:
                line=ff.readline()
                if line.find("BEGIN IONS")!=-1:
                    continue
                elif line.find("TITLE=")!=-1:
                    continue
                elif line.find("PEPMASS=")!=-1:
                    continue
                elif line.find("CHARGE=")!=-1:
                    continue
                elif line.find("SCANS=")!=-1:
                    continue
                elif line.find("RTINSECONDS=")!=-1:
                    ms2_rt=float(line[len("RTINSECONDS="):-1])
                elif line.find("END ION")!=-1:
                    break
                elif len(line)>3:
                    pp=line.strip().split(" ")
                    ms2_spect.append([float(pp[0]), float(pp[1]), ms2_rt])
                else:
                    print(line)
                    ff.close()
                    sys.exit(1)
        ff.close()
        fragments=[]
        if seq != "":
            tmp_seq=seq.replace("@","C").replace("~","M")
            for i in range(len(seq)):
                fragments.append(mass.fast_mass(tmp_seq[:i], ion_type='b', charge=int(z)) + 57.021 * seq[:i].count(conv_mod["C(+57.02)"])+15.99*seq[:i].count(conv_mod["M(+15.99)"]))
                fragments.append(mass.fast_mass(tmp_seq[i:], ion_type='y', charge=int(z)) + 57.021 * seq[i:].count(conv_mod["C(+57.02)"])+15.99*seq[i:].count(conv_mod["M(+15.99)"]))
            for peak in ms2_spect:
                peak_label = 0
                for frag in fragments:
                    if np.abs(peak[0] - frag) < (20 * frag / 1e6):
                        peak_label = 1
                        break
                fragment_label.append(peak_label)
            for _ in range(len(ms1_spect)):
                fragment_label.append(2)
            
            token_seq+=[self.index["$"]]
            rev_token_seq+=[self.index["$"]]
        
        ms2_spect=np.asarray(ms2_spect)
        ms1_n=sum(ms1_norm)
        ms1_spect=np.asarray(ms1_spect)
        ms1_spect[:,1]=ms1_spect[:,1]/ms1_n
        spectra=np.vstack((ms2_spect,ms1_spect))

        return (spectra, token_seq, rev_token_seq, fragment_label,mz,z)
###############################################


def collate_fn(batch:object):
    spectra=[]
    pep_tokens=[]
    rev_pep_token=[]
    fragment_label=[]
    mz=[]
    z=[]
    for tmp in batch:
        tmp_spectra, tmp_pep_tokens, tmp_rev_pep_token, f_label, tmp_mz, tmp_z = tmp
        spectra.append(torch.tensor(tmp_spectra))
        pep_tokens.append(torch.tensor(tmp_pep_tokens))
        rev_pep_token.append(torch.tensor(tmp_rev_pep_token))
        fragment_label.append(torch.tensor(f_label))
        mz.append(tmp_mz)
        z.append(tmp_z)
    
    spectra=torch.nn.utils.rnn.pad_sequence(spectra,batch_first=True)
    pep_tokens=torch.nn.utils.rnn.pad_sequence(pep_tokens,batch_first=True)
    rev_pep_token=torch.nn.utils.rnn.pad_sequence(rev_pep_token,batch_first=True)
    fragment_label = torch.nn.utils.rnn.pad_sequence(fragment_label, batch_first=True) 
    precursor_masses = (np.array(mz) - 1.007276) * np.array(z)
    precursors = torch.vstack([torch.tensor(precursor_masses), torch.tensor(z), torch.tensor(np.array(mz))])

    return spectra.type(model_type), pep_tokens, rev_pep_token, fragment_label, precursors.T.type(model_type)
