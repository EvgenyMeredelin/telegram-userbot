import asyncio

import pandas as pd
from decouple import config
from pyrogram import filters
from pyromod import Client


prompt = """Определи тему новостной статьи.

Возможные темы:
{labels}.

Выбирай только из возможных тем. В ответе укажи только тему.

Новостная статья:
{text}
"""


def classify(texts: list[str]) -> list[str]:
    # coroutine expects pyromod.Client and pyromod.Message as positional
    # arguments: https://github.com/usernein/pyromod#examples
    @client.on_message(filters.user(bot_name))
    async def ask_bot(*args, **kwargs) -> None:
        index, text = kwargs["index"], kwargs["text"]
        response = await client.ask(bot_name, text)
        labels[index] = response.text

    async def create_tasks():
        tasks = []
        for index, text in enumerate(texts):
            task = asyncio.create_task(ask_bot(index=index, text=text))
            tasks.append(task)
            # prevent pyrogram.errors.exceptions.flood_420.FloodWait
            await asyncio.sleep(20)
        await asyncio.wait(tasks)

    labels = [None] * len(texts)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(create_tasks())
    return labels


if __name__ == "__main__":
    client = Client(
        name=config("TELEGRAM_ACCOUNT_NAME"),
        api_id=config("TELEGRAM_API_ID"),
        api_hash=config("TELEGRAM_API_HASH"),
        phone_number=config("PHONE_NUMBER")
    )
    bot_name = config("TELEGRAM_BOT_NAME")

    news = (
        pd.read_csv("data/news.csv")
        .rename(columns={"rubric": "true_label"})
        .dropna(subset="true_label")
    )
    drop_labels = [
        "69-я параллель", "Бывший СССР", "Россия",
        "Мир", "Силовые структуры", "Нацпроекты"
    ]
    news = news[~news.true_label.isin(drop_labels)]
    sample = news.sample(50, random_state=42)
    valid_labels = sample.true_label.unique()
    texts = [
        prompt.format(
            text=text[:text.find(". ", 500) + 1],
            labels=", ".join(valid_labels)
        )
        for text in sample.text
    ]

    client.start()
    sample["ai_label"] = classify(texts)
    subset = ["text", "true_label", "ai_label"]
    sample[subset].to_csv("data/sample.csv", index=False)
    client.stop()

    accuracy = (sample.ai_label == sample.true_label).mean()
    print(accuracy)
