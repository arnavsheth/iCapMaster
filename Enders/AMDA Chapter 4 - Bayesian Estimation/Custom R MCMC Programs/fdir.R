#' Create a filename  with the current working directory
#'
#' @param directory A string.
#' @param filename  A string.
#' @return A string
#' @export

fdir <- function(directory,filename) {
  return(paste0(getwd(), "/", directory, "/", filename))
}