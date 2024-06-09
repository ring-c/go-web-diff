package generate

import (
	"github.com/labstack/echo/v4"
)

func Routes(e *echo.Echo) {
	var r = e.Group("/")
	r.GET("", Index)
	r.POST("generate", Generate)
	r.GET("output/:filename", OutputDir)
}
