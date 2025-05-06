
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import jarque_bera
from scipy.optimize import minimize
import matplotlib.ticker as ticker
# import seaborn as sns

from fredapi import Fred

### 인터넷 접속 라이브러리 (한국은행,통계청)
import requests



###########################################################
#     Py_Dashboard
###########################################################

### 여러개의 FRED 데이터를 한번에 입수

def multi_fred(api_key,ticker,observation_start,observation_end):

  # 라이브러리 정의
  fred = Fred(api_key = api_key)

  # 데이터 정리할 빈데이터프레임 정의
  fred_data = pd.DataFrame()

  # ticker 루프 실행
  for i,j in enumerate(ticker['fred']):

    # Fred 데이터 입수
    fred_data_ = fred.get_series(j, observation_start=observation_start, observation_end=observation_end)

    if i == 0 :
      fred_data = fred_data_

    if i > 0 :
      fred_data = pd.concat([fred_data,fred_data_],axis = 1)

  # 컬럼명 부여
  fred_data.columns = ticker['name']

  return fred_data

### 통계청 데이터 입수

def get_kosis(api_key,tables,observation_start,observastion_end,freq):

  ### url 기록할 빈 리스트 정의
  urls = [""] * len(tables)

  ### url에 데이터 주기, 조회기간 추가
  prdSe = freq
  startPrdDe = observation_start
  endPrdDe = observastion_end

  url ="https://kosis.kr/openapi/Param/statisticsParameterData.do?method=getList&apiKey=" + api_key + "&" \
       "format=json&jsonVD=Y&prdSe=" + prdSe + "&startPrdDe=" + startPrdDe + "&endPrdDe=" + endPrdDe

  ### url에 org_id,table_id,item_id,기타 조회조건 추가
  for i in range(0,len(tables)):

    ### org_id,table_id,item_id 추가
    urls[i] = url  + "&orgId=" + tables.loc[i,'org_ID'] + "&tblId=" + tables.loc[i,'tbl_ID'] + "&itmId=" + tables.loc[i,'item_ID']

    ### 기타 조회조건 추가
    for j in tables.columns:
      if j == 'obj_L1':
        urls[i] = urls[i] + "&objL1=" + tables.loc[i,j]
      if j == 'obj_L2':
        urls[i] = urls[i] + "&objL2=" + tables.loc[i,j]
      if j == 'obj_L3':
        urls[i] = urls[i] + "&objL3=" + tables.loc[i,j]
      if j == 'obj_L4':
        urls[i] = urls[i] + "&objL4=" + tables.loc[i,j]

  ### 데이터 입수
  df_kosis = pd.DataFrame()  # 빈 데이터테이블 정의

  for i in range(0,len(tables)):

    ### 데이터 입수
    response = requests.get(urls[i])
    data = response.json()
    df = pd.DataFrame(data)
    df_kosis = pd.concat([df_kosis,df])

  ### 입수한 데이터를 날자 * 컬럼 형태로 피벗

  # 입수된 데이터는 문자열 형태이므로 숫자로 변환
  df_kosis['DT'] = pd.to_numeric(df_kosis['DT'])

  # 컬럼갯수가 가변적이므로 C1_NM~C8_NM 중에서 데이터프레임 내에 있는 컬럼만 추출
  # 컬럼명을 모두 _로 연결하여 피벗테이블의 컬럼명으로 함
  col_nm = ['C1_NM','C2_NM','C3_NM','C4_NM','C5_NM','C6_NM','C7_NM','C8_NM']
  sel_col_nm = df_kosis.columns[df_kosis.columns.isin(col_nm)]
  df_kosis['COL_NM'] = df_kosis['TBL_NM'] + '_'
  for i in sel_col_nm:
    df_kosis['COL_NM'] = df_kosis['COL_NM'] + df_kosis[i] + '_'
  df_kosis['COL_NM'] = df_kosis['COL_NM'].str[:-1]

  # 피벗 테이블 작성 및 인덱스를 datetime 형태로 변환
  df_kosis_pivot = pd.pivot_table(data = df_kosis,index = ['PRD_DE'],columns = ['COL_NM'],values = 'DT')
  df_kosis_pivot.index = pd.to_datetime(df_kosis_pivot.index,format = '%Y%m')

  return df_kosis_pivot

    
### 한국은행 데이터 입수

def get_bok(api_key,data,freq,data_from,data_to):

  bok_out = pd.DataFrame()

  if freq == 'D':
    date_format = '%Y%m%d'
  if freq == 'M':
    date_format = '%Y%m'


  for i in range(0,len(data)):

    table_cd = data.loc[i,'table_cd']

    item_code_2 = '?'
    item_code_3 = '?'
    item_code_4 = '?'

    item_code_1 = data.loc[i,'item_code_1']
    item_code_2 = data.loc[i,'item_code_2']
    item_code_3 = data.loc[i,'item_code_3']
    item_code_4 = data.loc[i,'item_code_4']

    url = 'https://ecos.bok.or.kr/api/StatisticSearch/' + api_key + \
          '/JSON/kr/1/10000/'+table_cd+'/'+freq+'/' + data_from + '/' + data_to + '/' +\
          item_code_1+'/'+item_code_2+'/'+item_code_3+'/'+item_code_4

    response = requests.get(url)
    bok_data = response.json()
    df = pd.DataFrame(bok_data['StatisticSearch']['row'])
    df['SR_NAME'] = data.loc[i,'sr_name']
    bok_out = pd.concat([bok_out,df])

  bok_out['DATA_VALUE'] = pd.to_numeric(bok_out['DATA_VALUE'])
  bok_out['TIME'] = pd.to_datetime(bok_out['TIME'],format=date_format)
  bok_out_pivot = pd.pivot_table(data = bok_out,index = ['TIME'],columns = ['SR_NAME'],values = 'DATA_VALUE',aggfunc = 'mean')

  return bok_out_pivot



###########################################################
#     Py_Features of a financial Time Series
###########################################################

### 일별 가격데이터 => 일별,주별,월별 수익률

def get_return(p, freq='D'):

    """
    일별 가격 데이터를 받아서 일별,주별,월별 수익률을 산출
    인덱스는 날짜 형식일 것
    """
    p = p.sort_index()

    if freq == 'D':  # 일별 수익률
        r = p.pct_change().dropna(how='all')

    if freq == 'W':  # 주별 수익률 (금~금)
        p_week = p.resample('W-FRI').last().astype(float)  # 일별 데이터를 기간데이터(주별)로 변환하여 마지막 수치만 추출
        r = p_week.ffill().pct_change().dropna(how='all')
        
    if freq == 'M':  # 월별 수익률
        p_month = p.resample('ME').last().astype(float)  # 일별 데이터를 기간데이터(월별)로 변환하여 마지막 수치만 추출
        r = p_month.ffill().pct_change().dropna(how='all')

    return r

### 수익률 연율화

def annualize_return(r,periods_per_year):
    """
    periods_per_year : 월별 = 12, 주별 = 52, 일별 = 365 or 250
    """
    r = pd.DataFrame(r) # r이 시리즈인 경우에 데이터프레임으로 전환
    df = pd.DataFrame(columns = r.columns, index = ['Annual_Rtn'],dtype = 'float')

    for i in r.columns:
      
      n_period =r[i].dropna().shape[0]
      compounded_return = (1+r[i]).prod()
      df.loc['Annual_Rtn',i] = compounded_return**(periods_per_year / n_period) - 1

    return df


### 표준편차 연율화

def annualize_vol(r,periods_per_year):
    """
    periods_per_year : 월별 = 12, 주별 = 52, 일별 = 365 or 250
    """
    r = pd.DataFrame(r) # r이 시리즈인 경우에 데이터프레임으로 전환
    df = pd.DataFrame(columns = r.columns, index = ['Annual_Std'],dtype = 'float')

    for i in r.columns:
      df.loc['Annual_Std',i] = r[i].std()*(periods_per_year**0.5)

    return df

### 금융시계열 분포의 정규성 검정 및 정규분포와 비교 그래프

def check_normal(rtn,show_dist= True):

  """
  수익률 분포의 정규성 확인
  """

  # 결과 저장할 빈 데이터셋 정의
  rtn_desc = pd.DataFrame(columns = rtn.columns, index = ['Mean','Standard Deviation','Skewness','Kurtosis','JB_Stat','JB_P-Val'])

  # 정규성 검정 관련 통계량
  for i in rtn.columns:
    rtn_desc.loc['Mean',i] = rtn[i].mean()  # 평균
    rtn_desc.loc['Standard Deviation',i] = rtn[i].std() # 표준편차
    rtn_desc.loc['Skewness',i] = rtn[i].skew()  # 왜도
    rtn_desc.loc['Kurtosis',i] = rtn[i].kurtosis()  # 첨도
    jb_test = jarque_bera(rtn[i]) # Jarque-bera test statistics(정규성 검정)
    rtn_desc.loc['JB_Stat',i] = jb_test[0]
    rtn_desc.loc['JB_P-Val',i] = jb_test[1]

  # 그래프
  if show_dist:

    for i in rtn.columns:        

      z = 4

      rtn_mean = rtn[i].mean()
      rtn_std = rtn[i].std()

      lower_limit = rtn_mean - z * rtn_std
      upper_limit = rtn_mean + z * rtn_std

      x_range = np.linspace(lower_limit, upper_limit, 100)
      fig, ax = plt.subplots(figsize=(15, 5))

      # density = True : 히스토그램의 면적이 1이 되도록 함
      plt.hist(rtn[i], density=True, bins=100, label=i, color='blue') 
      plt.plot(x_range, norm.pdf(x_range, loc=rtn_mean, scale=rtn_std), label='Normal Dist.', color='orange',
                linewidth=3)

      for j in range(-3, 4):
          plt.axvline(rtn_mean + j * rtn_std, 0, 25, color='lightgray', linestyle='--', linewidth=2)

      plt.title(i)
      plt.xlabel('return', fontsize=12)
      plt.legend()
      plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

      plt.show()

  return rtn_desc

  ### Value at Risk 산출 함수 

def value_at_risk(rtn, level):

  """
  method = normal : Normal 분포 적용
  method = cf     : Conrish - Fisher expansion 적용
  method = hist   : Historical 분포 적용
  """

  df_VaR = pd.DataFrame(columns = rtn.columns,index = ['Normal VaR','C-F VaR','Historical VaR'])

  for i in rtn.columns:

    r = rtn[i].dropna()

    # Normal VaR
    z = norm.ppf(level/100) # Percent Point Function (norm.ppf(0.975) → 약 1.96)
    var_pct_normal = r.mean() + z * r.std()

    # Historical VaR
    var_pct_hist = np.percentile(r, level)

    # modify the Z score based on observed skewness and kurtosis
    s = r.skew()
    k = r.kurt()
    z_cf = (z +
            (z**2 - 1)*s/6 +
            (z**3 -3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
        )

    var_pct_cf = r.mean() + z_cf * r.std()

    df_VaR.loc['Normal VaR',i] = var_pct_normal
    df_VaR.loc['C-F VaR',i] = var_pct_cf
    df_VaR.loc['Historical VaR',i] = var_pct_hist

  return df_VaR

### MDD 산출 함수

def drawdown(p, show_mdd = False):

  """
  p의 전고점 대비 하락률
  """

  p = pd.DataFrame(p)   # p가 시리즈인 경우 데이터프레임으로 변환
  p_prev_max = pd.DataFrame(columns = p.columns)
  p_drawdown = pd.DataFrame(columns = p.columns)
  p_mdd = pd.DataFrame(columns = p.columns, index = ['MDD'])

  for i in p.columns:
    p_prev_max[i] = p[i].cummax()  # 전고점 값
    p_drawdown[i] = (p[i]-p_prev_max[i])/p_prev_max[i]  # 전고점 대비 현재가
    p_mdd.loc['MDD',i] = p_drawdown[i].min()

  if show_mdd == True:

    for i in p.columns:
      fig,ax = plt.subplots(1,2,figsize = (15,5))

      ax[0].plot(p[i],label = i)
      ax[0].plot(p_prev_max[i],label = 'prev_max')
      ax[1].plot(p_drawdown[i],label = 'Drawdown')
      ax[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))

      ax[0].legend()
      ax[1].legend()
      plt.suptitle(i)
      plt.show()

  return p_drawdown,p_mdd

### 연간 수익률,변동성, VaR, MDD, Sharpe 등 시계열의 특성 보여주는 함수

def fs_desc(p, freq, rf = 0.03):

  """
    연간수익률, 변동성, VaR, MDD, Sharpe 등 주요 통계량을 하나로 보여줌
  """

  r = pd.DataFrame(get_return(p,freq = freq))

  if freq == 'W':
    periods_per_year = 52

  if freq == 'M':
    periods_per_year = 12

  if freq == 'D':
    periods_per_year = 252


  r_rtn = annualize_return(r,periods_per_year)
  r_vol = annualize_vol(r,periods_per_year)
  r_var = value_at_risk(r,level=5)
  r_dd,r_mdd = drawdown(p,show_mdd = False)
  
  r_sharpe = pd.DataFrame(columns = r.columns, index = ['Sharpe'])
  r_sharpe.loc['Sharpe',:] = (r_rtn.loc['Annual_Rtn',:]-rf)/r_vol.loc['Annual_Std',:]
  r_all = pd.concat([r_rtn,r_vol,r_var,r_mdd,r_sharpe])

  return r_all

###########################################################
#     Py_Portfolio Management
###########################################################


def min_vol_target_return(target_return, er, cov):

    """
    :param target_return: 목표수익률
    :param er: 자산별 기대수익률
    :param cov: 포트폴리오 공분산
    :return: 최적화 결과
    """

    # 등식 제약 1: Sum of Weight = 1
    constraint_1 = {'type': 'eq',
                    'fun': lambda weight: np.sum(weight) - 1
                    }

    # 등식 제약 2: Target Return = Sum of (Weight * Expected Return)
    constraint_2 = {'type': 'eq',
                    'fun': lambda weight: target_return - weight @ er
                    }

    n_asset = er.shape[0]

    # 최적화를 위한 초기치 : 균등비율에서 출발
    init_guess = np.repeat(1 / n_asset, n_asset)

    # 각 자산의 비중은 0에서 1 사이
    bnds = ((0.0, 1.0),) * n_asset

    # 최소화 목적 함수 : min (sqrt(W COV W'))
    result = minimize(fun=lambda weight,cov: (weight @ cov @ weight.T)**0.5,
                      x0=init_guess,
                      args=(cov,),
                      method='SLSQP',
                      options={'disp': False},
                      constraints=(constraint_1, constraint_2),
                      bounds=bnds)

    return result.x


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights @ covmat @ weights.T)**0.5

### Sharpe Ratio 극대화

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x, weights.fun



### Risk Parity 자산배분 산출


def risk_contribution(wgt,vcv):

  pf_var = (wgt.T @ vcv @ wgt)
  rc1 = wgt.T * (vcv @ wgt)
  rc2 = rc1 / pf_var

  return rc2

def risk_parity(vcv):

    """
    risk parity (risk contribution의 sum square를 최소화)
    """
    constraint_1 = {'type': 'eq',
                    'fun': lambda weight: np.sum(weight) - 1
                    }

    n_asset = vcv.shape[0]
    er = np.repeat(1, n_asset)
    rf = 0.0
    init_guess = np.repeat(1/n_asset,n_asset)

    def min_squared_risk_contribution(weight, vcv):

        rc = risk_contribution(weight, vcv)
        trc =np.repeat(1/len(rc),len(rc))

        return np.sum((rc-trc)**2)

    bnds = ((0.0, 1.0),) * n_asset

    result = minimize(fun=min_squared_risk_contribution,
                      x0=init_guess,
                      args=vcv,
                      method='SLSQP',
                      options={'disp': False},
                      constraints=constraint_1,
                      bounds=bnds)

    return result.x

