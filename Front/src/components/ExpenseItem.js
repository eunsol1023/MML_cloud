import "./ExpenseItem.css";

function ExpenseItem() {
  //제목, 지출금액, 날짜 표시

  const expenseDate = new Date(2021, 2, 28); //자바코드
  const expenseTitle = "Car Insurance";
  const expenseAmount = 294.67;

  return (
    <div className="expense-item">
      <div>{expenseDate.toISOString()}</div>
      <div className="expense-item__description">
        <h2>{expenseTitle}</h2>
        <div className="expense-item__price">${expenseAmount}</div>
      </div>
    </div>
  );
}

export default ExpenseItem; 