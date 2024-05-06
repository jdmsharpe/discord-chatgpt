# Discord ChatGPT Bot

<!-- <div align="center">
  
[![HitCount](https://hits.dwyl.com/Nick-McGee/discord-bot.svg?style=flat)](http://hits.dwyl.com/Nick-McGee/discord-bot)
<a href="https://hub.docker.com/repository/docker/nrmcgee/discord-bot" target="_blank" rel="noopener noreferrer">![Workflow](https://github.com/Nick-McGee/discord-bot/actions/workflows/main.yml/badge.svg)</a>
  
</div> -->

## What is it?
This is a Discord bot built on <a href="https://github.com/Pycord-Development/pycord">Pycord 2.0</a>. It draws heavily <a href="https://github.com/Nick-McGee/discord-bot">Nick McGee's discord-bot</a>. The bot plays audio from a Youtube URL or search query. It is controlled with slash commands and the message-based user interface.

### Commands
+ **chat** - Query ChatGPT to generate text based on input

### UI

### Demo

## How to use it?
+ <a href="https://docs.pycord.dev/en/master/discord.html#:~:text=Make%20sure%20you're%20logged%20on%20to%20the%20Discord%20website.&text=Click%20on%20the%20%E2%80%9CNew%20Application,and%20clicking%20%E2%80%9CAdd%20Bot%E2%80%9D.">**Create a Discord Bot** and invite it to your Discord server</a>

### Build and Run with Docker (Recommended)
#### Build and run the image locally
+ Build the image with `docker build -t python-bot .` in the root directory
+ Run the bot with `docker run -e BOT_TOKEN=<YOUR BOT TOKEN> -e GUILD_ID=<YOUR GUILD IDS IN LIST FORMAT> python-bot` in the root directory

### Running from source
+ (Recommended) Create a virtual environment
+ Install the dependencies from `requirements.txt` with `pip install -r requirements.txt` in the root directory
+ Set an environment variable for BOT_TOKEN with your bot's token
+ Set an environment variable for GUILD_IDS with the Discord guild ids (servers) you wish to deploy the bot on
+ Run the bot with `python src/bot.py` in the root directory
