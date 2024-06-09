package generate

import (
	"github.com/labstack/echo/v4"
)

func OutputDir(c echo.Context) (err error) {
	return c.File("./output/" + c.Param("filename"))
}
