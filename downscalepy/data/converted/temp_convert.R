
load("/home/ubuntu/repos/downscalepy/downscalepy/data/original/argentina_raster.RData")

obj_name <- ls()[!ls() %in% c("input_file", "output_file")]

write.csv(get(obj_name), file="/home/ubuntu/repos/downscalepy/downscalepy/data/converted/argentina_raster.csv", row.names=FALSE)

cat("Columns:", paste(colnames(get(obj_name)), collapse=", "), "
")
