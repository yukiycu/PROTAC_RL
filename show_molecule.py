
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import Draw

def main():
    input_name = "case/dBET6/predictions_ms_2000_250.txt"
    output_path = "case/dBET6/"
    model_file = "/home/murakami/LabNote_Murakami/middle-molecule_permeability/result/lightgbm_mordred/train_only_protac/model_all.pkl"    
    fea_file = "/home/murakami/LabNote_Murakami/middle-molecule_permeability/result/feature/fea_mordred_list.pickle"
    show_molecule_num = 12
    with open(fea_file, mode='rb') as f:
        fea_list = pickle.load(f)
    with open(model_file, mode='rb') as f:
        model = pickle.load(f)
    f = open(input_name, 'r')
    data = f.readlines()
    f.close()
    mols = []
    error_smi_counter = 0
    for i in data:
        mol = Chem.MolFromSmiles(i)
        #正しい構造かどうかの判定
        isstandard = Chem.MolToSmiles(mol, isomericSmiles=True)
        if isstandard:
            smile = i.replace("\n", "")
            smile = smile.replace(" ", "")
            mol = Chem.MolFromSmiles(smile)
            try:
                Chem.MolToSmiles(mol, isomericSmiles=True)
                mols.append(mol)
            except:
                print("[ERROR] smiles error 1 : ", i)
                error_smi_counter += 1
                continue
        else:
            print("[ERROR] smiles error 2 : ", i)
            error_smi_counter += 1
    print("error_smi_counter: ", error_smi_counter)

    mols = set(mols)
    smiles2 = [Chem.MolToSmiles(mol) for mol in mols]
    print("num of molecule: ", len(mols))
    smiles3 = list(set(smiles2))
    mols2 = [Chem.MolFromSmiles(smile) for smile in smiles3]
    print("num of molecule: ", len(mols2))
    
    calc = Calculator(descriptors, ignore_3D=False)
    permeability = []
    for mol in mols2:
        mordred = calc.pandas([mol])
        df_mordred = pd.DataFrame(mordred)
        X = df_mordred[fea_list]
        X = np.array(X)
        y_pred = model.predict(X)
        permeability.append(y_pred[0])
    
    df_data = pd.DataFrame({'mols': mols2, 'permeability': permeability})
    sns.histplot(permeability,kde=True)
    plt.title("Result_Papp("+str(len(mols2))+")", fontsize=20)
    plt.xlabel("Papp (log10(μcm/s))", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.savefig(output_path+"result_papp_histplot.png", bbox_inches='tight')
    plt.close()
    
    df_data = df_data.sort_values('permeability', ascending=False)
    options = Draw.MolDrawOptions()
    options.legendFontSize = 20
    options.legendFraction = 0.2
    top_molecule = df_data.head(show_molecule_num)
    legend = [str(round(float(x), 3)) for x in top_molecule['permeability']]
    Draw.MolsToGridImage(top_molecule["mols"], molsPerRow=4, subImgSize=(300,300), legends=legend, drawOptions=options).save(output_path+'molecule.png')
    img = Draw.MolsToGridImage(top_molecule["mols"], molsPerRow=4, subImgSize=(300,300), legends=legend, useSVG=True, drawOptions=options)
    
    with open(output_path+'molecule.svg', 'w') as f:
        f.write(img)
    
    print("num of molecule: ", len(df_data))
    print('[INFO] Finished!')

if __name__ == '__main__':
    main()
