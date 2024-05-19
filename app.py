import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cities = ['Ahmedabad', 'Chennai', 'Mumbai', 'Bengaluru', 'Kolkata', 'Delhi',
       'Dharamsala', 'Hyderabad', 'Lucknow', 'Jaipur']

teams=['Gujarat Titans', 'Mumbai Indians', 'Chennai Super Kings',
      'Sunrisers Hyderabad', 'Royal Challengers Bangalore',
       'Lucknow Super Giants', 'Punjab Kings', 'Delhi Capitals',
       'Kolkata Knight Riders', 'Rajasthan Royals']

match = pd.read_csv('match_info_data.csv')
delivery = pd.read_csv('match_data.csv',low_memory=False)
pd.set_option('display.max_columns', None)

pd.options.mode.copy_on_write = True
match['city'] = match['city'].replace('Bangalore','Bengaluru')
match['city'] = match['city'].replace('Chandigarh','Dharamsala')
match['city'] = match['city'].replace('Indore','Dharamsala')
match['city'] = match['city'].replace('Rajkot','Ahmedabad')
match = match[match['city'].isin(cities)]

match_fields = ['team1', 'team2', 'toss_winner', 'winner']
delivery_fields = ['batting_team', 'bowling_team']

teams_mapping = {
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Gujarat Lions': 'Gujarat Titans'
}

for team in match_fields:
    match[team] = match[team].replace(teams_mapping)

for team in delivery_fields:
    delivery[team] = delivery[team].replace(teams_mapping)


match =match[match['team1'].isin(teams)]
match =match[match['team2'].isin(teams)]

delivery =delivery[delivery['batting_team'].isin(teams)]
delivery =delivery[delivery['bowling_team'].isin(teams)]

delivery.fillna(0, inplace=True)
delivery['total_runs'] = delivery['runs_off_bat']+delivery['extras']
delivery['wicket_type'] = delivery['wicket_type'].apply(lambda x: 1 if x != 0 else x)

match = match[match['dl_applied']==0]
delivery1 = delivery
delivery1 = delivery1.groupby(['match_id','innings'])['total_runs'].sum().reset_index()

delivery1 = delivery1[delivery1['innings']==1]

match = match.merge(delivery1[['match_id','total_runs']],left_on='id',right_on='match_id')
match = match[['match_id','city','winner','total_runs']]

delivery = match.merge(delivery,on='match_id')
delivery = delivery[delivery['innings'] == 2]
delivery['total_runs_y'] = pd.to_numeric(delivery['total_runs_y'], errors='coerce')
delivery['current_score'] = delivery.groupby('match_id')['total_runs_y'].cumsum()
delivery['runs_left'] = delivery['total_runs_x'] - delivery['current_score'] + 1
delivery['over'] = delivery['ball'].apply(int)
delivery['balls'] = (delivery['ball']-delivery['over'])*10
delivery['balls_left'] = 120 - (delivery['over']*6 + delivery['balls'])
delivery['wickets'] =delivery['wicket_type']+delivery['other_wicket_type']
delivery['wickets_taken'] = delivery.groupby('match_id')['wickets'].cumsum()
delivery['wickets_left']= 10 - delivery['wickets_taken']
delivery['crr'] = (delivery['current_score']*6)/(delivery['over']*6 + delivery['balls'])
delivery['rrr'] = (delivery['runs_left']*6)/delivery['balls_left']
def result(df):
    return 1 if df['batting_team'] == df['winner'] else 0
delivery['result'] = delivery.apply(result,axis=1)
final_df = delivery[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_x','crr','rrr','result']]
final_df= final_df[final_df['balls_left'] > 0]
final_df= final_df[final_df['runs_left'] > 0]
final_df = final_df.sample(final_df.shape[0])







X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)

accuracy_score(y_test,y_pred)

pickle.dump(pipe,open('pipe.pkl','wb'))




pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictor')


batting_team = st.selectbox('Select the batting team',sorted(teams))
bowling_team = st.selectbox('Select the bowling team',sorted(teams))
selected_city = st.selectbox('Select host city',sorted(cities))
target = st.number_input('Target')
score = st.number_input('Score')
overs = st.number_input('Overs completed')
wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")