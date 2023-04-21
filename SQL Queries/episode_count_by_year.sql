SELECT date_part('year', "pubDate") AS episode_year,
       count(*) as yearly_count
FROM episodes
GROUP BY episode_year
order by yearly_count desc;
