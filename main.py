import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

oil_price = pd.read_csv("data/oil_future_price.csv")
oil_price.date_time =oil_price.date_time.apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
oil_price = oil_price.sort_values(by="date_time", ascending=True)
oil_price.set_index('date_time', inplace=True)
oil_price['daily_log_return'] = np.log(oil_price.close) - np.log(oil_price.close.shift(1))
oil_price['daily_simple_return'] = oil_price.close.pct_change()


# sns.displot(oil_price, x="daily_log_return", kind="kde")

qqplot(oil_price.daily_simple_return[1:], line='q', dist=stats.norm, fit=True)