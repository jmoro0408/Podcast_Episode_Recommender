ALTER TABLE episodes ALTER COLUMN "pubDate" TYPE DATE 
using to_date("pubDate", 'Dy, DD Mon YYYY');
