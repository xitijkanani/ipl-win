# Win Probability Predictor

This is a web application that predicts the win probability for two cricket teams based on various inputs such as the batting team, bowling team, runs scored by the bowling team(Target), current score, wickets taken, city, and overs completed. 
The application uses Streamlit for the frontend and a machine learning model for making predictions.

## Features

- User Input: Enter details about the batting team, bowling team, target, current score, wickets taken, city, and overs completed.
- Win Probability Prediction: Get real-time win probabilities for both teams based on the provided inputs.
- Interactive Interface: User-friendly interface built with Streamlit for easy interaction.

## Installation

To run this application locally, you need to have Python installed on your machine. Follow these steps to set up and run the application:

1. Clone the repository:

    ```
    git clone https://github.com/xitijkanani/ipl-win-predictor.git
    cd ipl-win-predictor
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```
    streamlit run app.py
    ```

## Usage

1. Open the application:
   After running the above command, the application will open in your default web browser. If it doesn't, you can manually open it by navigating to `http://localhost:8501` in your browser.

2. Enter the input details:
   - Batting Team : Select the batting team from the dropdown menu.
   - Bowling Team : Select the bowling team from the dropdown menu.
   - Target : Enter the total runs scored by the bowling team.
   - Current Score : Enter the current score of the batting team.
   - Wickets Taken : Enter the number of wickets taken by the bowling team.
   - City : Enter the city where the match is being played.
   - Overs Completed : Enter the number of overs completed.

3. Get the Prediction:
   Click on the "Predict Probability" button to get the win probability for both teams.

## Example

Here's a quick example of how to use the app:

1. Select "Team A" as the batting team.
2. Select "Team B" as the bowling team.
3. Enter `150` for the Target.
4. Enter `120` for the current score.
5. Enter `3` for the wickets taken.
6. Enter `Mumbai` for the city.
7. Enter `15.0` for the overs completed.
8. Click "Predict Probability" to see the win probabilities.

## Dependencies

- Python 3.11.*
- Streamlit
- Pandas
- Scikit-learn (or other machine learning library you used)
- Any other dependencies listed in `requirements.txt`

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature-name`).
6. Open a pull request.

## Contact

If you have any questions or feedback, feel free to contact me at [kananixitij@gmail.com].

