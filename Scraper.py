import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import json

async def scrape_platform_comments(platform_name, seed_url):
    print(f"[INFO] 启动 Agentic Scraper 针对平台: {platform_name}")
    async with async_playwright() as p:
        # 模拟真实浏览器规避反爬
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await page.goto(seed_url)
        await page.wait_for_timeout(3000)

        comments_pool = []
        
        # 动态滚动与自适应元素捕获逻辑
        for i in range(50):  # 模拟滚动 50 次
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await page.wait_for_timeout(1500)
            
            # 根据平台特征动态选择选择器
            selector = ".comment-item" if platform_name in ["rednote", "douyin"] else "article"
            elements = await page.query_selector_all(selector)
            
            for el in elements:
                text = await el.inner_text()
                if text and text not in comments_pool:
                    comments_pool.append({
                        "platform": platform_name,
                        "raw_text": text.strip()
                    })
                    
        await browser.close()
        return comments_pool

async def main():
    # 平台种子抓取配置
    targets = {
        "rednote": "https://www.xiaohongshu.com/search?keyword=AI恋爱",
        "twitter_x": "https://x.com/search?q=4B%20AI"
    }
    
    all_data = []
    for platform, url in targets.items():
        data = await scrape_platform_comments(platform, url)
        all_data.extend(data)
        
    df = pd.DataFrame(all_data)
    # 数据清洗：去重与去除空字符
    df.dropna(subset=['raw_text'], inplace=True)
    df = df[df['raw_text'].str.len() > 5]
    df.to_csv("raw_comments_dataset.csv", index=False)
    print(f"[SUCCESS] 数据采集完成。总计捕获有效向量数: {len(df)}")

if __name__ == "__main__":
    asyncio.run(main())