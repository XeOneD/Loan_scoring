Automated Loan Approval System

The bot for approval the loan to users based on their personal information. Bot uses the prediction of regression model. User can't get the loan twice, the system is prevent such action.

<img width="1128" height="1052" alt="image" src="https://github.com/user-attachments/assets/d909243b-e187-4ba8-84de-92a30b8131b2" />
<img width="146" height="182" alt="image" src="https://github.com/user-attachments/assets/0b8ce9b3-c990-4359-9a82-c72acda4e69c" />
<img width="536" height="344" alt="image" src="https://github.com/user-attachments/assets/0b0f43f2-ca6c-4004-b859-f367ddf5ae65" />

Product context:  
•	End-user: Credit departments of microfinance organizations or banks seeking to automate the initial processing of applications.  
•	Problem Solved: High time costs for manual scoring of clients and the risk of human error in assessing solvency.  
•	Product Idea: An intelligent Telegram bot that instantly predicts the probability of loan approval based on the ML model and saves the application history in a reliable database.  

Features:  
•	How it was built:  
o Backend: Python (Aiogram 3.x) for the logic of the bot.  
o	Machine Learning: A logistic regression model (Scikit-learn) uploaded via Joblib.  
o	Database: PostgreSQL for storing user data.  
o	Infrastructure: Containerization via Docker Compose (3 services: Bot, DB, pgAdmin).  
•	Version 1 vs Version 2:  
o	V1: A simple bot using SQLite (a local file) and basic polling.  
o	V2: The transition to the industrial PostgreSQL DBMS, the introduction of an anti-fraud system (checking the telegram_id in the database) and deployment to a remote virtual machine (VM).  

Usage:  
•	"We launch the bot, press the Menu button."  
•	"We take a survey: enter the first name, last name, age, income, education and car."  
•	"The bot accesses the ML model and issues a verdict."  
•	"We show the browser with the pgAdmin open: the user's data instantly appeared in in the applications table."  
•	"We try to press /start again — the bot recognizes us and does not allow us to fill out the questionnaire again."  

Deployment:  
The VM should run on the Ubuntu 24.04  
On the VM should be cloned the repopsitory with all the required stuff  

1. Clone repository  
2. Go to reposetory on the VM  
3. Use this command to run contaners with the bot: sudo docker-compose up --build -d  
4. Check that containers are Up: sudo docker ps  
5. Use this command to stop contaners with the bot: sudo docker-compose down  
6. Open the bot in Telegram: @XeOneD_bot  
7. Start using the bot  
8. Open http://10.93.26.97:8080 in browser  
9. Use login: admin@admin.com and password: adminpassword to enter the pgAdmin  
10. Create a connection with given parameters:  
  Host name/address: db  
  Username: admin  
  Password: password  
