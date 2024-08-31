package generate

import (
	"net/http"

	"github.com/labstack/echo/v4"
)

func Routes(e *echo.Echo) {
	var r = e.Group("/")
	r.GET("", Index)
	r.POST("generate", Generate)
	r.GET("output/:filename", OutputDir)
	r.GET("models", Models)
}

func OutputDir(c echo.Context) (err error) {
	return c.File("./output/" + c.Param("filename"))
}

func Models(c echo.Context) (err error) {

	return c.JSON(http.StatusOK, "")
}
