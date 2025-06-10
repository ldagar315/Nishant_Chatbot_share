# CD-chatbot-v1
This is the repo shared with Nishant, to get started with the chatbot of country delight

## File Name and constituents
- App.py - Streamlit frontend for the code
- faq.md - Modified FAQs scraped from the Country Delight website, to answer general queries about the app and company
- system_prompt.py - The in depth system prompt for the chatbot, detailing each action
- main.py - Backend written in Langgraph, with all of the functions tools
- fake_db_creator.py - Creates Fake grocery database created successfully with 10 users, 50 products, and orders for 7 days.

## How the project work ?
First run fake_db_creator script to create a fake sql database of orders, then directly to go the app.py and in the terminal type streamlit run app.py, and the frontend would run, if not please check the api keys, and insert your own. 

## scope of the agent ?
The agent can handle many tasks on its own like 
- answering general queries 
- Issue a refund (for damaged, undelivered any other problems with order)
- analyse damaged products
- reply in voice mode 
We have made the agent to handle very basic queries. 

## API keys needed to run
- Deepgram (for voice)
- Groq / Cerebras (for the LLM calls)

## Libraries needed 
- Groq, langgraph, streamlit, sqlite3, langchain-cerebras, langchain-groq
