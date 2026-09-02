import pandas as pd
import numpy as np
import imputer.imputer as imputer
from utils import load_data

class Imputation:
    
    def __init__(self, df, config):
        self.data = df
        self.config = config

    def short_process(self, df):
        # 각 열마다 처리
        for idx in range(df.shape[1]):
            # 복사본 생성
            temp_df = df.iloc[:, idx].copy().to_frame()
            
            # 연속된 결측치의 개수 계산
            temp_df['nan_streak'] = temp_df.iloc[:, 0].isnull().astype(int).groupby(temp_df.iloc[:, 0].notnull().astype(int).cumsum()).cumsum()
            
            # 연속된 결측치 그룹 번호 생성
            temp_df['group_num'] = temp_df.iloc[:, 0].notnull().astype(int).cumsum()
            
            # 12개 이상의 연속된 결측치를 inf로 채우기
            for i in temp_df['group_num'].unique():
                if temp_df[temp_df['group_num']==i]['nan_streak'].max() >= 12:
                    temp_df.loc[temp_df['group_num']==i, temp_df.columns[0]] = temp_df.loc[temp_df['group_num']==i, temp_df.columns[0]].fillna(float('inf'))
            
            # 처리가 끝난 열을 원래 데이터프레임에 저장
            df.iloc[:, idx] = temp_df.iloc[:, 0]
            df = getattr(imputer, "interpolated")(df)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df
    def increase_number_in_string(s):
        # 문자열의 앞부분(문자)과 뒷부분(숫자)을 분리합니다.
        prefix = s[:-2]
        number = s[-2:]

        # 숫자 부분을 1 증가시킵니다.
        increased = int(number) + 1

        # 증가시킨 숫자가 10 미만이면 앞에 '0'을 붙여줍니다.
        if increased < 10:
            increased = '0' + str(increased)
        else:
            increased = str(increased)

        # 문자 부분과 증가시킨 숫자 부분을 합칩니다.
        return prefix + increased        
    
    def run(self):
        print("Processing...")
        df = self.short_process(self.data)
        
        if self.config['imputation'] == 'mice':
            p = self.config['query_proccess']['place']
            t = self.config['query_proccess']['typeno']
            result_df = df
            high_corr_count = 0
            ##################수정할 부분#################
            while True:
                for _ in range(17):
                    p = self.increase_number_in_string(p)
                    self.config['query_proccess']['place'] = p
                    for __ in range(16):
                        t = self.increase_number_in_string(t)
                        self.config['query_proccess']['place'] = t

                        df_ = load_data(self.config)
                        #이후로 상관분석
                        concat_df = pd.concat([df, df_], axis=1)
                        corr_matrix = concat_df.corr()

                        if abs(corr_matrix.loc[df.columns, df_.columns]).mean().mean() >= 0.8:
                        # 상관계수가 높다면 결과 데이터프레임에 해당 데이터프레임을 추가합니다.
                            result_df = pd.concat([result_df, df], axis=1)
                            high_corr_count += 1

                            if high_corr_count >= 10:
                                break
            ##############################################
        else:
            imputed = self.config.init_obj(self.config["imputation"], imputer, df)
        print("="*20)
        print("Number of missing data after process :\n",imputed.isna().sum())
        self.data = imputed


