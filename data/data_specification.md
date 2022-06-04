# 数据说明

（本数据的源头来自 李菁老师数据集 https://github.com/polyusmart/HEC-Dataset )


1. HAndRawLabels.csv 包含全部用户emoji投票信息
2. HLA_46.csv 包含hashtag/导语（abstract）/以及emoji投票信息，其中 "abstract_df_idx"列对应评论文件（e.g. CICD_merge_data）索引
3. Read_Comments.csv 为论文 Chp5 实验所用经筛选足量评论文本后的数据集
4. keywords_3_24.csv 为李菁老师团队数据基础上的 TextRank(window = 3)选前50关键词；
   keywords_CICD.csv 在带标签的评论（i.e. CICD_merge_data ）下，TextRank 选前100关键词
   
