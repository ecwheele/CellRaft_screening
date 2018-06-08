import pandas as pd


to_drop = ['AreaOccupied_Perimeter_Nuclei','AreaOccupied_Perimeter_Stress_granules',
           'AreaOccupied_TotalArea_Nuclei','AreaOccupied_TotalArea_Stress_granules',
           'FileName_GFP','Threshold_FinalThreshold_Nuclei',
           'Threshold_FinalThreshold_Stress_granules',
           'Threshold_OrigThreshold_Nuclei',
           'Threshold_OrigThreshold_Stress_granules',
           'Threshold_SumOfEntropies_Nuclei',
           'Threshold_SumOfEntropies_Stress_granules',
           'Threshold_WeightedVariance_Nuclei',
           'Threshold_WeightedVariance_Stress_granules',
           'ImageNumber',
           'FileName_cellbody',
           'URL_DAPI',
           'URL_GFP',
           'URL_cellbody']


old = ['AreaOccupied_AreaOccupied_Nuclei','AreaOccupied_AreaOccupied_Stress_granules']
new = ['Nuc_area','SG_area']
rename_dict = dict(zip(old,new))


def get_wellid(df):
    df['wellid'] = df['FileName_DAPI'].apply(lambda x: x.split("blue.tiff")[0])
    return df


def calculate_sg_dapi_area(df):
    order = ['Nuc_area', 'SG_area', 'SG/DAPI_area', 'Count_Nuclei', 'Count_Stress_granules',
       'wellid','FileName_DAPI']
    df['SG/DAPI_area'] = zoom2['SG_area']/zoom2['Nuc_area']
    df = df[order]
    return df.sort_values(by="SG/DAPI_area")


def read_in_cellprofiler(filename, to_drop=to_drop, rename_dict=rename_dict):
    """
    reads in and processes csv output from cellprofiler
    :param file: csv file
    :param to_drop:
    :param rename_dict:
    :return:
    """
    df = pd.read_csv(filename)
    df.drop(to_drop, axis=1, inplace=True)
    df.rename(columns = rename_dict, inplace=True)
    df = get_wellid(df)
    df = calculate_sg_dapi_area(df)
    return df







